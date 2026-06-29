"""
popularity_baseline.py
-----------------------
Tính baseline "luôn gợi ý N item phổ biến nhất toàn cục" (không cá nhân hoá
gì cả) trên CÙNG test_df / cold_start_users để so sánh trực tiếp, công bằng,
với kết quả CFLS. Nếu CFLS không vượt được con số này, nghĩa là toàn bộ phần
"học cá nhân hoá" trong pipeline hiện tại chưa mang lại giá trị thực, và
hướng đầu tư nên chuyển sang: làm giàu feature/profile, tăng dữ liệu, hoặc
đổi kiến trúc — không nên tiếp tục chỉnh hyperparameter của GP/VAE nữa.

Cách dùng: copy hàm popularity_topk_metrics(...) vào ngay sau khi
Evaluator.evaluate_model(...) chạy xong trong train.py (cả val và test),
truyền vào interaction_df/sparse_mat đã build sẵn.
"""
import numpy as np


def popularity_topk_metrics(interaction_df, test_df, cold_start_users, top_k_list):
    print("\n" + "=" * 70)
    print("BASELINE: POPULARITY (khong ca nhan hoa)")
    print("=" * 70)

    # Item phổ biến nhất = tổng số lượt nghe (đếm dòng) trong interaction_df,
    # đây chính là nguồn dữ liệu CFLS dùng để xây s_u/SVD -> so sánh công bằng.
    pop_counts = interaction_df["track_idx"].value_counts()
    k_max = max(top_k_list)
    top_items_global = pop_counts.index.values[:k_max]   # đã sort giảm dần

    test_users  = test_df["user_idx"].values.astype(np.int64)
    test_tracks = test_df["track_idx"].values.astype(np.int64)
    cold_set = set(int(u) for u in cold_start_users)
    cold_mask = np.array([int(u) in cold_set for u in test_users])
    warm_mask = ~cold_mask

    # MỌI user nhận đúng 1 danh sách top_items_global giống nhau
    top_k_mat = np.tile(top_items_global, (len(test_users), 1)).astype(np.int32)
    actual = test_tracks.astype(np.int32)

    def _metrics(top_k_matrix, actual_array, k_list):
        import math
        out = {}
        for k in k_list:
            top_k = top_k_matrix[:, :k]
            hits = (top_k == actual_array[:, None])
            hit_any = hits.any(axis=1)
            ranks = np.arange(1, k + 1)[None, :]
            rr = np.where(hits, 1.0 / ranks, 0.0).max(axis=1)
            ndcg = np.where(hits, math.log(2) / np.log2(ranks + 1), 0.0).max(axis=1)
            out[k] = {
                "precision": float(hit_any.mean() / k),
                "recall": float(hit_any.mean()),
                "map": float(rr.mean()),
                "ndcg": float(ndcg.mean()),
            }
        return out

    all_m  = _metrics(top_k_mat, actual, top_k_list)
    cold_m = _metrics(top_k_mat[cold_mask], actual[cold_mask], top_k_list) if cold_mask.any() else None
    warm_m = _metrics(top_k_mat[warm_mask], actual[warm_mask], top_k_list) if warm_mask.any() else None

    def _print(name, m):
        if m is None:
            return
        print(f"\n--- {name} ---")
        for k in top_k_list:
            print(f"  K={k:<3}  P={m[k]['precision']*100:.4f}%  R={m[k]['recall']*100:.4f}%  "
                  f"MAP={m[k]['map']*100:.4f}%  NDCG={m[k]['ndcg']*100:.4f}%")

    _print("TAT CA USERS (Popularity baseline)", all_m)
    _print("COLD-START (Popularity baseline)", cold_m)
    _print("WARM (Popularity baseline)", warm_m)

    print("\n=> So sanh NDCG@20 nay voi NDCG@20 cua CFLS de biet model co thuc su")
    print("   hoc duoc gia tri ca nhan hoa nao vuot qua 'doan theo do pho bien' khong.")
    return {"all": all_m, "cold": cold_m, "warm": warm_m}


if __name__ == "__main__":
    print(__doc__)