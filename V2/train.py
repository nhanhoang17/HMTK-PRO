import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pprint

from config import Config
from data_prep import DataPipeline
from embeddings import EmbeddingEngine, FactorizationMachine
from cfls_model import CFLS
from evaluate import Evaluator

'''
def elbo_loss(s_u, s_hat, z_u, logvar, X_profile, model):
    B, M = s_u.shape
    _, Kd = z_u.shape
    sigma_s_sq = model.sigma_s_sq

    # --- reconstruction term, chuan hoa ve "trung binh moi phan tu (B*M)" ---
    log_term = torch.log(sigma_s_sq)
    sse_term = torch.mean((s_u - s_hat) ** 2) / (2.0 * sigma_s_sq)
    recon    = log_term + sse_term

    # --- latent-space GP term: -log p(Z_psi | X, theta, alpha) ---
    K = model.compute_gp_covariance(X_profile)
    try:
        L_c = torch.linalg.cholesky(K)
    except RuntimeError:
        K   = K + 1e-4 * torch.eye(B, device=K.device)
        L_c = torch.linalg.cholesky(K)

    log_det   = 2.0 * torch.log(torch.diagonal(L_c)).sum()
    quad_form = (z_u * torch.cholesky_solve(z_u, L_c)).sum()
    gp_lp     = -0.5 * (Kd * log_det + quad_form + B * Kd * math.log(2 * math.pi))
    gp_term_raw = -gp_lp
    gp_term = gp_term_raw / (B * Kd)

    # --- regularization term, chuan hoa ve "trung binh moi phan tu (B*K)" ---
    reg_raw = 0.5 * logvar.sum()
    reg     = reg_raw / (B * Kd)

    loss = recon + 0.3 * (gp_term - reg)

    return loss, {
        "recon":      recon.item(),
        "recon_log":  log_term.item(),
        "recon_sse":  sse_term.item(),
        "gp":         gp_term.item(),
        "reg":        reg.item(),
        "total":      loss.item(),
        "alpha":      model.alpha.item(),
        "sigma_f":    model.sigma_f.item(),
        "sigma_s_sq": sigma_s_sq.item(),
    }
'''
def elbo_loss(s_u, s_hat, z_u, logvar, X_profile, model):
    B, M = s_u.shape
    _, Kd = z_u.shape
    sigma_s_sq = model.sigma_s_sq

    # 1. RECONSTRUCTION TERM (Sửa cốt lõi ở đây)
    # SUM theo chiều M (giữ sức mạnh 128 chiều), MEAN theo Batch B
    log_term = (M / 2.0) * torch.log(sigma_s_sq)
    sse_term = torch.sum((s_u - s_hat) ** 2, dim=1).mean() / (2.0 * sigma_s_sq)
    recon    = log_term + sse_term

    # 2. GP TERM
    K = model.compute_gp_covariance(X_profile)
    try:
        L_c = torch.linalg.cholesky(K)
    except RuntimeError:
        K   = K + 1e-4 * torch.eye(B, device=K.device)
        L_c = torch.linalg.cholesky(K)

    log_det = 2.0 * torch.log(torch.diagonal(L_c)).sum()
    
    # Giữ nguyên cách tính quad_form cũ của bạn. Ví dụ:
    quad_form = (z_u * torch.cholesky_solve(z_u, L_c)).sum()
    
    # [QUAN TRỌNG]: Nhớ chia GP term cho B để đồng bộ "Mean per batch"
    gp_term = 0.5 * (Kd * log_det + quad_form) / B

    # 3. REG TERM
    # Tương tự, SUM theo chiều Kd (64), MEAN theo Batch B
    reg = 0.5 * torch.sum(logvar, dim=1).mean()

    # 4. TỔNG LOSS
    beta = 0.05 
    loss = recon + beta * (gp_term - reg)

    return loss, {
        "recon":      recon.item(),
        "recon_log":  log_term.item(),
        "recon_sse":  sse_term.item(),
        "gp":         (beta * gp_term).item(),
        "reg":        (beta * reg).item(),
        "total":      loss.item(),
        "alpha":      model.alpha.item(),
        "sigma_f":    model.sigma_f.item(),
        "sigma_s_sq": sigma_s_sq.item(),
    }

def main():
    print(f"=== CFLS PIPELINE (Device: {Config.DEVICE}) ===\n")

    pipeline = DataPipeline()
    df = pipeline.load()

    # Nhan output tu DataPipeline moi (9 bien, co val_cold_users va test_cold_users)
    interaction_df, val_df, test_df, train_users, val_users, test_users, val_cold_users, test_cold_users, all_users_profile_df = pipeline.process_and_split(df)

    sparse_mat = pipeline.build_interaction_matrix(interaction_df)
    item_emb = EmbeddingEngine.get_svd_item_embeddings(sparse_mat, Config.M_DIM).to(Config.DEVICE)
    item_emb = item_emb * 10.0

    fm_model = FactorizationMachine(
        num_tz    = len(pipeline.tz_enc.classes_),
        num_lang  = len(pipeline.lang_enc.classes_),
        num_tlang = len(pipeline.tlang_enc.classes_),
        num_items = pipeline.n_items,
        embed_dim = Config.FM_EMBED_DIM,
    ).to(Config.DEVICE)
    try:
        fm_model.load_state_dict(torch.load(Config.FM_WEIGHTS_PATH, map_location=Config.DEVICE))
    except FileNotFoundError:
        pass
    fm_model.eval()

    n_users = pipeline.n_users

    # 1. X_profile CHO TOÀN BỘ USERS (Dùng cho nội suy Cold-start)
    all_users_df = all_users_profile_df.sort_values("user_idx")
    X_tmp        = fm_model.extract_user_profiles(all_users_df).numpy()
    all_X_np     = np.zeros((n_users, Config.FM_EMBED_DIM * 3), dtype=np.float32)
    all_X_np[all_users_df["user_idx"].values] = X_tmp

    # [THÊM 2 DÒNG NÀY]: Chuẩn hóa Z-score cho X_profile
    X_mean = all_X_np.mean(axis=0, keepdims=True)
    X_std  = all_X_np.std(axis=0, keepdims=True) + 1e-8
    all_X_np = (all_X_np - X_mean) / X_std

    all_X_profile = torch.tensor(all_X_np)

    # 2. Vector s_u CHO TOÀN BỘ USERS
    s_u_full = EmbeddingEngine.prepare_su_vectors(sparse_mat, item_emb.cpu())

    # =====================================================================
    # TẬP TRAIN KHÔNG THAY ĐỔI: Chỉ lấy 70% train_users đưa vào DataLoader
    # =====================================================================
    """
    train_idxs_all = torch.tensor(train_users, dtype=torch.long)
    train_X_all    = all_X_profile[train_idxs_all].to(Config.DEVICE)
    train_su_all   = s_u_full[train_idxs_all].to(Config.DEVICE)

    nz_mask       = train_su_all.any(dim=1)
    train_su_nz   = train_su_all[nz_mask]
    train_X_nz    = train_X_all[nz_mask]
    train_idx_nz  = train_idxs_all[nz_mask]
    """
    nz_mask       = s_u_full.any(dim=1)
    train_su_nz   = s_u_full[nz_mask].to(Config.DEVICE)
    train_X_nz    = all_X_profile[nz_mask].to(Config.DEVICE)
    train_idx_nz  = torch.arange(n_users)[nz_mask]

    #print(f"Base Train users: {nz_mask.sum().item()} / {len(train_su_all)}")
    print(f"Base Train users: {nz_mask.sum().item()} / {len(s_u_full)}")

    dl = DataLoader(TensorDataset(train_su_nz, train_X_nz), batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)

    model = CFLS(Config.M_DIM, Config.K_DIM, Config.FM_EMBED_DIM * 3).to(Config.DEVICE)
    #-------------------------------------------------------------------------------------
    from diagnose_collapse import run_all_diagnostics
    
    run_all_diagnostics(
        model         = model,
        fm_model      = fm_model,
        weights_path  = Config.FM_WEIGHTS_PATH,
        s_u_full      = s_u_full,
        nz_mask       = nz_mask,
        all_X_profile = all_X_profile,
        train_X_nz    = train_X_nz,
        device        = Config.DEVICE,
    )
    #-------------------------------------------------------------------------------------
    hyperparam_names = {"log_sigma_f", "log_alpha", "log_tau", "log_sigma_s_sq"}
    hyperparam_params = [p for n, p in model.named_parameters() if n in hyperparam_names]
    network_params    = [p for n, p in model.named_parameters() if n not in hyperparam_names]

    HYPERPARAM_LR_SCALE = getattr(Config, "HYPERPARAM_LR_SCALE", 0.1)
    opt = optim.Adam([
        {"params": network_params,    "lr": Config.LR},
        {"params": hyperparam_params, "lr": Config.LR * HYPERPARAM_LR_SCALE},
    ])

    HYPERPARAM_WARMUP_EPOCHS = getattr(Config, "HYPERPARAM_WARMUP_EPOCHS", 5)

    print("\n--- Huan luyen CFLS ---")
    best_val_ndcg, wait, best_state = -float("inf"), 0, None

    for epoch in range(Config.EPOCHS_CFLS):
        model.train()
        model.set_hyperparam_grad(epoch >= HYPERPARAM_WARMUP_EPOCHS)

        tot = {"recon": 0., "recon_log": 0., "recon_sse": 0., "gp": 0., "reg": 0., "total": 0.}
        nb  = 0
        for bs_u, bX in dl:
            opt.zero_grad()
            s_hat, mu, logvar, z_u = model(bs_u)
            loss, parts = elbo_loss(bs_u, s_hat, z_u, logvar, bX, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k in tot: tot[k] += parts[k]
            nb += 1

        avg = {k: v / nb for k, v in tot.items()}
        warmup_tag = " [warmup]" if epoch < HYPERPARAM_WARMUP_EPOCHS else ""
        print(f"Epoch [{epoch+1:2d}/{Config.EPOCHS_CFLS}]{warmup_tag} "
              f"total={avg['total']:9.2f} | recon={avg['recon']:9.2f} "
              f"(log={avg['recon_log']:9.2f}, sse={avg['recon_sse']:8.6f}) | "
              f"gp={avg['gp']:9.2f} | reg={avg['reg']:8.2f} | "
              f"sigma_s_sq={model.sigma_s_sq.item():.4f} | "
              f"sigma_f={model.sigma_f.item():.4f} | alpha={model.alpha.item():.4f}")

        # TẬP VAL (10% users): Kiểm tra Early Stopping
        print("  [Validation]")
        val_results = Evaluator.evaluate_model(
            model            = model,
            train_su         = train_su_nz,       
            train_X          = train_X_nz,
            train_idxs       = train_idx_nz.numpy(),
            test_df          = val_df,            
            cold_start_users = val_cold_users,    # <-- Truyền val_cold_users vào đây
            all_X_profile    = all_X_profile.to(Config.DEVICE),
            all_su           = s_u_full,
            item_embeddings  = item_emb,
            top_k_list       = [20],
            max_support      = Config.SUPPORT_SIZE,
            n_samples        = 1,
            batch_size       = Config.EVAL_BATCH_SIZE,
        )

        val_ndcg = val_results["all"][20]["ndcg"]
        print(f"  -> Validation NDCG@20 = {val_ndcg:.4f} (Tong hop tu Warm va Cold)")

        if val_ndcg > best_val_ndcg:
            best_val_ndcg, wait = val_ndcg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= Config.PATIENCE:
                print(f"Early stopping tai epoch {epoch+1}.")
                break

    if best_state: model.load_state_dict(best_state)
    print(f"Best Val NDCG@20: {best_val_ndcg:.4f}")

    print(f"\n--- Danh gia TEST CUOI CUNG (r={Config.COLD_START_RATIO}) ---")
    results = Evaluator.evaluate_model(
        model            = model,
        train_su         = train_su_nz,
        train_X          = train_X_nz,
        train_idxs       = train_idx_nz.numpy(),
        test_df          = test_df,
        cold_start_users = test_cold_users,  # <-- Truyền test_cold_users vào đây
        all_X_profile    = all_X_profile.to(Config.DEVICE),
        all_su           = s_u_full,          
        item_embeddings  = item_emb,
        top_k_list       = Config.TOP_K,
        max_support      = Config.SUPPORT_SIZE,
        n_samples        = Config.N_SAMPLES,
        batch_size       = Config.EVAL_BATCH_SIZE,
    )
    print("\n===== KET QUA CUOI =====")
    pprint.pprint(results)

    from popularity_baseline import popularity_topk_metrics

    popularity_topk_metrics(
        interaction_df   = interaction_df,
        test_df          = test_df,
        cold_start_users = test_cold_users,
        top_k_list       = Config.TOP_K,   # [20, 40, 60, 80]
    )

    run_all_diagnostics(
        model         = model,
        fm_model      = fm_model,
        weights_path  = Config.FM_WEIGHTS_PATH,
        s_u_full      = s_u_full,
        nz_mask       = nz_mask,
        all_X_profile = all_X_profile,
        train_X_nz    = train_X_nz,
        device        = Config.DEVICE,
    )

if __name__ == "__main__":
    main()