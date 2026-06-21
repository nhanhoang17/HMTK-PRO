"""
train.py
--------
FIX: them beta de can bang reconstruction va GP term.

Van de: recon=11 << gp=5500
-> Model bi dominated boi GP term
-> Encoder khong hoc duoc representation tot

Giai phap: nhan recon voi he so beta > 1 de can bang:
    L = beta * recon + gp_term - reg

beta = M (chieu song embedding) la mot lua chon pho bien:
-> recon/M ~ gp/B*K -> 2 term co cung order of magnitude

Luu y: day la approximation, paper khong co beta
nhung viec sigma_s_sq != 1 trong thuc te tuong duong them beta.
"""

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


def elbo_loss(s_u, s_hat, z_u, logvar, X_profile, model, beta=1.0):
    """
    L = beta * sum||s_u - s_hat||^2 / 2
      + gp_term
      - reg

    beta: he so can bang recon va gp (mac dinh 1.0 = paper goc)
    Tang beta khi recon << gp de encoder hoc tot hon.
    """
    B, M = s_u.shape

    recon = beta * torch.sum((s_u - s_hat)**2) / 2.0

    K = model.compute_gp_covariance(X_profile)
    try:
        L_c = torch.linalg.cholesky(K)
    except RuntimeError:
        K   = K + 1e-4 * torch.eye(B, device=K.device)
        L_c = torch.linalg.cholesky(K)

    log_det   = 2.0 * torch.log(torch.diagonal(L_c)).sum()
    quad_form = (z_u * torch.cholesky_solve(z_u, L_c)).sum()
    _, Kd     = z_u.shape
    gp_lp     = -0.5 * (Kd*log_det + quad_form + B*Kd*math.log(2*math.pi))
    gp_term   = -gp_lp

    reg  = 0.5 * logvar.sum()
    loss = recon + gp_term - reg

    return loss, {
        "recon":   recon.item(),
        "gp":      gp_term.item(),
        "reg":     reg.item(),
        "total":   loss.item(),
        "alpha":   model.alpha.item(),
        "sigma_f": model.sigma_f.item(),
    }


def main():
    print(f"=== CFLS PIPELINE (Device: {Config.DEVICE}) ===\n")

    # 1. Du lieu
    pipeline = DataPipeline()
    df = pipeline.load()
    train_df, test_df, cold_start_users = pipeline.process_and_split(df)
    sparse_mat = pipeline.build_interaction_matrix(train_df)

    # 2. Song embeddings
    item_emb = EmbeddingEngine.get_svd_item_embeddings(
        sparse_mat, Config.M_DIM).to(Config.DEVICE)

    # 3. FM user profiles
    fm_model = FactorizationMachine(
        len(pipeline.tz_enc.classes_),
        len(pipeline.lang_enc.classes_),
        len(pipeline.tlang_enc.classes_),
        Config.FM_EMBED_DIM,
    ).to(Config.DEVICE)
    try:
        fm_model.load_state_dict(
            torch.load(Config.FM_WEIGHTS_PATH, map_location=Config.DEVICE))
        print("Da nap FM weights.")
    except FileNotFoundError:
        print("CANH BAO: Khong co FM weights.")
    fm_model.eval()

    # Profile tat ca users
    n_users      = pipeline.n_users
    all_users_df = (train_df.drop_duplicates("user_idx")
                             .sort_values("user_idx"))
    X_tmp        = fm_model.extract_user_profiles(all_users_df).numpy()
    all_X_np     = np.zeros((n_users, Config.FM_EMBED_DIM*3), dtype=np.float32)
    all_X_np[all_users_df["user_idx"].values] = X_tmp
    all_X_profile = torch.tensor(all_X_np)

    train_idxs_all = torch.tensor(
        all_users_df["user_idx"].values, dtype=torch.long)
    train_X_all    = all_X_profile[train_idxs_all].to(Config.DEVICE)
    s_u_full       = EmbeddingEngine.prepare_su_vectors(sparse_mat, item_emb.cpu())
    train_su_all   = s_u_full[train_idxs_all].to(Config.DEVICE)

    nz_mask       = train_su_all.any(dim=1)
    train_su_nz   = train_su_all[nz_mask]
    train_X_nz    = train_X_all[nz_mask]
    train_idx_nz  = train_idxs_all[nz_mask]
    print(f"Train users co su!=0: {nz_mask.sum().item()} / {len(train_su_all)}")

    # He so can bang recon va GP
    # recon_scale = gp_scale khi beta ~ B*K_DIM / M_DIM
    beta = Config.M_DIM  # 128: nhan recon len ngang GP
    print(f"Beta (recon scale): {beta}")

    dl = DataLoader(TensorDataset(train_su_nz, train_X_nz),
                    batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)

    model = CFLS(Config.M_DIM, Config.K_DIM,
                 Config.FM_EMBED_DIM*3).to(Config.DEVICE)
    opt   = optim.Adam(model.parameters(), lr=Config.LR)

    print("\n--- Huan luyen CFLS ---")
    best_loss, wait, best_state = float("inf"), 0, None

    for epoch in range(Config.EPOCHS_CFLS):
        model.train()
        tot = {"recon":0., "gp":0., "reg":0., "total":0.}
        nb  = 0
        for bs_u, bX in dl:
            opt.zero_grad()
            s_hat, mu, logvar, z_u = model(bs_u)
            loss, parts = elbo_loss(bs_u, s_hat, z_u, logvar, bX, model, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k in tot: tot[k] += parts[k]
            nb += 1

        avg = {k: v/nb for k,v in tot.items()}
        print(f"Epoch [{epoch+1:2d}/{Config.EPOCHS_CFLS}] "
              f"total={avg['total']:9.2f} | recon={avg['recon']:9.2f} | "
              f"gp={avg['gp']:9.2f} | reg={avg['reg']:8.2f} | "
              f"alpha={parts['alpha']:.4f} | sigma_f={parts['sigma_f']:.4f}")

        if avg["total"] < best_loss:
            best_loss, wait = avg["total"], 0
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= Config.PATIENCE:
                print(f"Early stopping tai epoch {epoch+1}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"Best loss: {best_loss:.2f}")

    print(f"\n--- Danh gia (r={Config.COLD_START_RATIO}) ---")
    results = Evaluator.evaluate_model(
        model            = model,
        train_su         = train_su_nz,
        train_X          = train_X_nz,
        train_idxs       = train_idx_nz.numpy(),
        test_df          = test_df,
        cold_start_users = cold_start_users,
        all_X_profile    = all_X_profile.to(Config.DEVICE),
        item_embeddings  = item_emb,
        top_k_list       = Config.TOP_K,
        max_support      = Config.SUPPORT_SIZE,
        n_samples        = Config.N_SAMPLES,
        batch_size       = Config.EVAL_BATCH_SIZE,
    )
    print("\n===== KET QUA CUOI =====")
    pprint.pprint(results)


if __name__ == "__main__":
    main()