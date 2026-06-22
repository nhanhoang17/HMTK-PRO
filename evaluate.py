"""
evaluate.py
-----------
Dung GP posterior cho TAT CA users (ca warm lan cold-start).

Ly do (theo paper Sect. 3.4):
  Paper mo ta MOT quy trinh duy nhat cho moi user u*:
    (i)   Encode train users -> Z_psi* (support set)
    (ii)  Sample z_u* tu GP posterior p(z_u*| x_u*, Z_psi*, X)
    (iii) Decode z_u* -> s_hat_u*
    (iv)  Top-Q nearest neighbor

  GP tu dong xu ly ca 2 truong hop:
  - WARM user: z_psi_u co trong support set
               -> K(x_u, x_u) lon -> mu* ~ z_psi_u -> ket qua giong encoder
               -> Var* nho -> on dinh
  - COLD user: khong co z_psi_u trong support set
               -> mu* = weighted avg cua z_psi users co profile tuong tu
               -> Var* lon hon -> da dang hon (diversity)

Toi uu toc do:
  - Top-K: batch matmul + argpartition (9x nhanh hon loop argsort)
  - Metrics: vectorized (khong dung iterrows)
"""

import math
import numpy as np
import torch


class Evaluator:

    # ------------------------------------------------------------------
    # GP prediction (dung cho TAT CA users)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_support(model, train_su, train_X, max_support, device):
        """
        Tao support set tu train users.
        Encode train_su -> Z_psi (support).
        Neu qua lon thi sample ngau nhien max_support users.
        """
        n = train_su.size(0)
        if n > max_support:
            idx      = torch.randperm(n)[:max_support]
            sup_su   = train_su[idx]
            sup_X    = train_X[idx]
        else:
            sup_su, sup_X = train_su, train_X

        with torch.no_grad():
            mu_train, _ = model.encode(sup_su)   # (n_sup, K)

        return mu_train, sup_X

    @staticmethod
    def _gp_posterior(model, mu_train, sup_X, test_X, n_samples, device):
        """
        Tinh GP posterior cho test_X, sample n_samples lan, decode, average.

        Cong thuc:
            mu*   = K(X*, X_sup) @ K_sup^{-1} @ Z_psi
            var*  = diag(K(X*,X*) - K(X*,X_sup) @ K_sup^{-1} @ K(X_sup,X*))
            z*    ~ N(mu*, var*)  [sample n_samples lan]
            s_hat = mean(decoder(z*))

        Returns: s_hat (n_test, M) numpy array
        """
        with torch.no_grad():
            K_sup = model.compute_gp_covariance(sup_X)
            try:
                L = torch.linalg.cholesky(K_sup)
            except RuntimeError:
                K_sup += 1e-4 * torch.eye(K_sup.size(0), device=device)
                L     = torch.linalg.cholesky(K_sup)

            # Cache K_sup^{-1} @ Z_psi
            y       = torch.linalg.solve_triangular(L, mu_train, upper=False)
            K_inv_Z = torch.linalg.solve_triangular(L.T, y, upper=True)

            # K(X*, X_sup)
            K_star  = model.compute_cross_covariance(test_X, sup_X)  # (n_test, n_sup)

            # Posterior mean: (n_test, K)
            mu_star = K_star @ K_inv_Z

            # Posterior variance (diagonal only)
            v        = torch.linalg.solve_triangular(L, K_star.T, upper=False)
            prior_var = model.sigma_f**2 + model.alpha   # scalar
            var_diag  = (prior_var * torch.ones(
                test_X.size(0), device=device)
             - (v**2).sum(dim=0)).clamp(min=1e-6)
            std_star = var_diag.sqrt()   # (n_test,)

            # Sample -> decode -> average
            s_hat_sum = torch.zeros(test_X.size(0), model.m_dim, device=device)
            for _ in range(n_samples):
                eps   = torch.randn_like(mu_star)
                z_smp = mu_star + std_star.unsqueeze(1) * eps
                s_hat_sum += model.decoder(z_smp)

        return (s_hat_sum / n_samples).cpu().numpy()   # (n_test, M)

    # ------------------------------------------------------------------
    # Top-K: batch matmul + argpartition (tranh OOM, nhanh)
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_top_k(s_hat_np, item_np, k_max, batch_size):
        """
        Tinh top-K items cho tat ca users theo batch.
        RAM peak: batch_size * n_items * 4 bytes (vd: 512 * 173k * 4 = 354 MB).
        """
        n_users      = s_hat_np.shape[0]
        top_k_matrix = np.zeros((n_users, k_max), dtype=np.int32)

        for start in range(0, n_users, batch_size):
            end    = min(start + batch_size, n_users)
            scores = s_hat_np[start:end] @ item_np.T          # (B, n_items)
            part   = np.argpartition(scores, -k_max, axis=1)[:, -k_max:]
            for i in range(end - start):
                row   = scores[i, part[i]]
                order = np.argsort(row)[::-1]
                top_k_matrix[start + i] = part[i][order]

        return top_k_matrix

    # ------------------------------------------------------------------
    # Metrics vectorized
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics(top_k_matrix, actual_array, k_list):
        """
        Precision, Recall, MAP (=MRR), NDCG cho tat ca users cung luc.
        Voi leave-one-out: moi user co dung 1 relevant item.
        """
        out = {}
        for k in k_list:
            top_k = top_k_matrix[:, :k]                    # (n, k)
            hits  = (top_k == actual_array[:, None])        # (n, k)

            hit_any  = hits.any(axis=1)
            ranks    = np.arange(1, k+1)[None, :]           # (1, k)
            rr       = np.where(hits, 1.0/ranks, 0.0).max(axis=1)
            ndcg     = np.where(hits,
                                math.log(2)/np.log2(ranks+1),
                                0.0).max(axis=1)
            out[k] = {
                "precision": float(hit_any.astype(float).mean() / k),
                "recall":    float(hit_any.astype(float).mean()),
                "map":       float(rr.mean()),
                "ndcg":      float(ndcg.mean()),
            }
        return out

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_model(model,
                       train_su,
                       train_X,
                       train_idxs,
                       test_df,
                       cold_start_users,
                       all_X_profile,
                       item_embeddings,
                       top_k_list,
                       max_support,
                       n_samples=5,
                       batch_size=512):
        """
        Evaluate theo paper: GP posterior cho TAT CA users.

        Warm user: z_psi_u co trong support set -> mu* ~ z_psi_u (on dinh)
        Cold user: z_psi_u khong trong support -> mu* tu CF (da dang)

        Parameters
        ----------
        train_su    : (n_tr, M) su cua train users (co su != 0)
        train_X     : (n_tr, L) xu tuong ung
        train_idxs  : (n_tr,)  user_idx tuong ung
        all_X_profile: (n_all_users, L) xu cua TAT CA users
        """
        print(f"Danh gia (GP cho tat ca users) ...")
        print(f"  support<={max_support} | samples={n_samples} | batch={batch_size}")
        model.eval()
        device  = next(model.parameters()).device
        item_np = item_embeddings.cpu().numpy()
        k_max   = max(top_k_list)
        cold_set = set(int(u) for u in cold_start_users)

        # ── (i) Xay dung support set ──────────────────────────────────
        mu_train, sup_X = Evaluator._build_support(
            model, train_su, train_X, max_support, device)
        print(f"  Support set: {mu_train.shape[0]} users")

        # ── (ii) GP posterior cho TAT CA test users ───────────────────
        # Lay xu cua tat ca test users (ca warm lan cold)
        test_users  = test_df["user_idx"].values.astype(np.int64)
        test_tracks = test_df["track_idx"].values.astype(np.int64)

        # all_X_profile da co xu cho tat ca users (ke ca cold-start)
        test_X_all = all_X_profile[test_users].to(device)   # (n_test, L)

        print(f"  Tinh GP posterior cho {len(test_users)} users ...")
        s_hat_all = Evaluator._gp_posterior(
            model, mu_train, sup_X, test_X_all, n_samples, device)
        # (n_test, M)

        # ── (iii) Top-K ───────────────────────────────────────────────
        print(f"  Tinh Top-{k_max} (batch={batch_size}) ...")
        top_k_mat = Evaluator._batch_top_k(s_hat_all, item_np, k_max, batch_size)

        # ── (iv) Metrics ──────────────────────────────────────────────
        actual = test_tracks.astype(np.int32)

        is_cold = np.array([int(u) in cold_set for u in test_users])
        is_warm = ~is_cold

        all_metrics  = Evaluator._metrics(top_k_mat,           actual,        top_k_list)
        cold_metrics = Evaluator._metrics(top_k_mat[is_cold],  actual[is_cold], top_k_list) \
                       if is_cold.any() else _zero_metrics(top_k_list)
        warm_metrics = Evaluator._metrics(top_k_mat[is_warm],  actual[is_warm], top_k_list) \
                       if is_warm.any() else _zero_metrics(top_k_list)

        final = {"all": all_metrics, "cold": cold_metrics, "warm": warm_metrics}

        n_cold_eval = is_cold.sum()
        n_warm_eval = is_warm.sum()
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
        hdr = f"{'Metric':<12}" + "".join(f"  K={k:<8}" for k in top_k_list)
        print(hdr + "\n" + "-"*len(hdr))
        for m in ["precision", "recall", "map", "ndcg"]:
            row = f"{m.upper()+'@K':<12}" + "".join(
                f"  {results[k][m]*100:.2f}%  " for k in top_k_list)
            print(row)


def _zero_metrics(top_k_list):
    return {k: {"precision":0., "recall":0., "map":0., "ndcg":0.}
            for k in top_k_list}
