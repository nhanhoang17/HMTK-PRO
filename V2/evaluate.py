import math
import numpy as np
import torch

class Evaluator:
    @staticmethod
    def _build_support(model, train_su, train_X, max_support, device):
        n = train_su.size(0)
        if n > max_support:
            idx      = torch.randperm(n)[:max_support]
            sup_su   = train_su[idx]
            sup_X    = train_X[idx]
        else:
            sup_su, sup_X = train_su, train_X

        with torch.no_grad():
            mu_train, _ = model.encode(sup_su)
        return mu_train, sup_X

    @staticmethod
    def _gp_posterior(model, mu_train, sup_X, test_X, n_samples, device):
        with torch.no_grad():
            K_sup = model.compute_gp_covariance(sup_X)
            try:
                L = torch.linalg.cholesky(K_sup)
            except RuntimeError:
                K_sup += 1e-4 * torch.eye(K_sup.size(0), device=device)
                L     = torch.linalg.cholesky(K_sup)

            y       = torch.linalg.solve_triangular(L, mu_train, upper=False)
            K_inv_Z = torch.linalg.solve_triangular(L.T, y, upper=True)

            K_star  = model.compute_cross_covariance(test_X, sup_X)  
            mu_star = K_star @ K_inv_Z

            v        = torch.linalg.solve_triangular(L, K_star.T, upper=False)
            var_diag = (model.sigma_f**2 * torch.ones(test_X.size(0), device=device)
                        - (v**2).sum(dim=0)).clamp(min=1e-6)
            std_star = var_diag.sqrt()   
            s_hat_sum = torch.zeros(test_X.size(0), model.m_dim, device=device)
            
            for _ in range(n_samples):
                eps   = torch.randn_like(mu_star)
                z_smp = mu_star + std_star.unsqueeze(1) * eps
                s_hat_sum += model.decoder(z_smp)

        return (s_hat_sum / n_samples).cpu().numpy() 
    
    @staticmethod
    def _batch_top_k(s_hat_np, item_np, k_max, batch_size):
        n_users      = s_hat_np.shape[0]
        top_k_matrix = np.zeros((n_users, k_max), dtype=np.int32)

        for start in range(0, n_users, batch_size):
            end    = min(start + batch_size, n_users)
            scores = s_hat_np[start:end] @ item_np.T        
            
            part   = np.argpartition(scores, -k_max, axis=1)[:, -k_max:]
            for i in range(end - start):
                row   = scores[i, part[i]]
                order = np.argsort(row)[::-1]
                top_k_matrix[start + i] = part[i][order]

        return top_k_matrix
    
    @staticmethod
    def _metrics(top_k_matrix, actual_array, k_list):
        out = {}
        for k in k_list:
            top_k = top_k_matrix[:, :k]                     
            hits  = (top_k == actual_array[:, None])        

            hit_any  = hits.any(axis=1)
            ranks    = np.arange(1, k+1)[None, :]           
            rr       = np.where(hits, 1.0 / ranks, 0.0).max(axis=1)
            ndcg     = np.where(hits, math.log(2) / np.log2(ranks + 1), 0.0).max(axis=1)
            
            out[k] = {
                "precision": float(hit_any.astype(float).mean() / k),
                "recall":    float(hit_any.astype(float).mean()),
                "map":       float(rr.mean()),
                "ndcg":      float(ndcg.mean()),
            }
        return out
    
    @staticmethod
    def evaluate_model(model,
                       train_su,
                       train_X,
                       train_idxs,
                       test_df,
                       cold_start_users,
                       all_X_profile,
                       all_su,             # <-- THÊM THAM SỐ NÀY ĐỂ WARM USER DÙNG
                       item_embeddings,
                       top_k_list,
                       max_support,
                       n_samples=5,
                       batch_size=512):
        
        """
        Evaluate phan luong (Routing):
        - WARM: dung mang Encoder (chinh xac, nhanh)
        - COLD: dung GP Posterior (noi suy)
        """
        print(f"Danh gia (Phan luong Warm/Cold) ...")
        print(f"  support<={max_support} | samples={n_samples} | batch={batch_size}")
        model.eval()
        device  = next(model.parameters()).device
        item_np = item_embeddings.cpu().numpy()
        k_max   = max(top_k_list)
        cold_set = set(int(u) for u in cold_start_users)

        # ── (i) Xay dung support set cho GP ───────────────────────────
        mu_train, sup_X = Evaluator._build_support(
            model, train_su, train_X, max_support, device)
        print(f"  Support set: {mu_train.shape[0]} users")

        # ── (ii) Phan luong Du doan (Routing) ─────────────────────────
        test_users  = test_df["user_idx"].values.astype(np.int64)
        test_tracks = test_df["track_idx"].values.astype(np.int64)

        cold_mask = np.array([int(u) in cold_set for u in test_users])
        warm_mask = ~cold_mask

        # Khoi tao ma tran ket qua tong
        s_hat_all = np.zeros((len(test_users), model.m_dim), dtype=np.float32)

        # 1. Xu ly nhom COLD-START (Dung Gaussian Process)
        if cold_mask.any():
            print(f"  Tinh GP cho {cold_mask.sum()} COLD users ...")
            test_X_cold = all_X_profile[test_users[cold_mask]].to(device)
            s_hat_cold  = Evaluator._gp_posterior(
                model, mu_train, sup_X, test_X_cold, n_samples, device)
            s_hat_all[cold_mask] = s_hat_cold

        # 2. Xu ly nhom WARM (Dung truc tiep Mang Encoder)
        if warm_mask.any():
            print(f"  Encode truc tiep cho {warm_mask.sum()} WARM users ...")
            warm_user_idxs = test_users[warm_mask]
            warm_su = all_su[warm_user_idxs].to(device)
            
            with torch.no_grad():
                mu_warm, _ = model.encode(warm_su)
                s_hat_warm = model.decoder(mu_warm).cpu().numpy()
            s_hat_all[warm_mask] = s_hat_warm
        
        # ── (ii.5) DIAGNOSTIC: do do da dang du doan giua cac user ─────
        def _diversity_report(s_hat, mask, name):
            sub = s_hat[mask]
            if len(sub) < 2: return
            sub_norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
            sim = sub_norm @ sub_norm.T
            avg_pairwise_sim = (sim.sum() - len(sub)) / (len(sub)*(len(sub)-1))
            print(f"  [DIVERSITY] {name}: avg pairwise cosine sim giua cac s_hat = {avg_pairwise_sim:.4f}")

        _diversity_report(s_hat_all, cold_mask, "COLD")
        _diversity_report(s_hat_all, warm_mask, "WARM")

        # ── (iii) Top-K ───────────────────────────────────────────────
        print(f"  Tinh Top-{k_max} (batch={batch_size}) ...")
        top_k_mat = Evaluator._batch_top_k(s_hat_all, item_np, k_max, batch_size)

        # ── (iv) Metrics ──────────────────────────────────────────────
        actual = test_tracks.astype(np.int32)

        all_metrics  = Evaluator._metrics(top_k_mat,           actual,          top_k_list)
        cold_metrics = Evaluator._metrics(top_k_mat[cold_mask], actual[cold_mask], top_k_list) \
                       if cold_mask.any() else _zero_metrics(top_k_list)
        warm_metrics = Evaluator._metrics(top_k_mat[warm_mask], actual[warm_mask], top_k_list) \
                       if warm_mask.any() else _zero_metrics(top_k_list)

        final = {"all": all_metrics, "cold": cold_metrics, "warm": warm_metrics}

        n_cold_eval = cold_mask.sum()
        n_warm_eval = warm_mask.sum()
        print(f"\n===== TAT CA USERS ({len(test_users)}) =====")
        Evaluator._print_table(final["all"], top_k_list)
        if n_cold_eval > 0:
            print(f"\n===== COLD-START ({n_cold_eval}) =====")
            Evaluator._print_table(final["cold"], top_k_list)
        if n_warm_eval > 0:
            print(f"\n===== WARM ({n_warm_eval}) =====")
            Evaluator._print_table(final["warm"], top_k_list)

        return final
    
    @staticmethod
    def _print_table(results, top_k_list):
        hdr = f"{'Chỉ số':<12}" + "".join(f"  K={k:<8}" for k in top_k_list)
        print(hdr + "\n" + "-"*len(hdr))
        for m in ["precision", "recall", "map", "ndcg"]:
            row = f"{m.upper()+'@K':<12}" + "".join(
                f"  {results[k][m]*100:.2f}%  " for k in top_k_list)
            print(row)

def _zero_metrics(top_k_list):
    return {k: {"precision": 0., "recall": 0., "map": 0., "ndcg": 0.} for k in top_k_list}
#-----------------------------------------------------------------------------------------------------#



