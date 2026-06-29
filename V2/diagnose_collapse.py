"""
diagnose_collapse.py
---------------------
Script CHẨN ĐOÁN (không train lại từ đầu) — chạy SAU khi bạn đã có:
  - sparse_mat, item_emb (SVD)
  - fm_model (đã load hoặc chưa)
  - s_u_full, all_X_profile
  - model CFLS (có thể là model đã train hoặc model mới init)

Cách dùng: copy đoạn dưới vào ngay sau bước build xong các biến trong train.py
(sau dòng `s_u_full = EmbeddingEngine.prepare_su_vectors(...)`), hoặc import
hàm `run_all_diagnostics(...)` và gọi trực tiếp.

Mỗi hàm in ra kết luận PASS/FAIL/WARNING rõ ràng cho từng giả thuyết.
"""

import os
import numpy as np
import torch


# --------------------------------------------------------------------------- #
def _cosine_diversity(mat: np.ndarray, max_n=3000, seed=0):
    """avg pairwise cosine similarity — giống _diversity_report trong evaluate.py"""
    if len(mat) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(mat), size=max_n, replace=False)
        mat = mat[idx]
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    n = len(mat)
    return (sim.sum() - n) / (n * (n - 1))


def _verdict(ok, msg_ok, msg_fail):
    print(("  [PASS] " + msg_ok) if ok else ("  [FAIL] " + msg_fail))
    return ok


# --------------------------------------------------------------------------- #
# GT 1: FM weights có thực sự được load không, hay đang ở random init?
# --------------------------------------------------------------------------- #
def check_fm_weights_loaded(fm_model, weights_path):
    print("\n[GT-1] FM weights có được load thành công không?")
    exists = os.path.isfile(weights_path)
    print(f"  File tồn tại tại '{weights_path}': {exists}")
    if not exists:
        print("  [FAIL] FILE KHÔNG TỒN TẠI -> fm_model đang ở RANDOM INIT (std=0.01)."
              " Đây gần như chắc chắn là nguyên nhân chính.")
        return False

    # So sánh embedding hiện tại với 1 bản random init mới để xem có "giống random" không
    with torch.no_grad():
        cur_std = fm_model.emb_tz.weight.std().item()
        cur_mean_abs = fm_model.emb_tz.weight.abs().mean().item()
    print(f"  emb_tz.weight std={cur_std:.5f} | mean(abs)={cur_mean_abs:.5f}")
    ok = cur_std > 0.02  # random init std=0.01; nếu train xong std nên lớn hơn rõ rệt
    return _verdict(
        ok,
        f"std={cur_std:.4f} > 0.02, embedding có vẻ đã được học (khác random init std=0.01).",
        f"std={cur_std:.4f} ~ random init (0.01) -> NGHI VẤN: weight chưa được train/load đúng "
        f"(kiểm tra lại đường dẫn FM_WEIGHTS_PATH và việc train_fm() đã chạy & lưu thành công chưa).",
    )


# --------------------------------------------------------------------------- #
# GT 2: X_profile (đầu vào GP) có đủ phương sai/đa dạng giữa user không?
# --------------------------------------------------------------------------- #
def check_profile_diversity(all_X_profile: torch.Tensor, user_idxs=None):
    print("\n[GT-2] X_profile (FM-derived) có phân biệt được user không?")
    X = all_X_profile
    if user_idxs is not None:
        X = X[user_idxs]
    X_np = X.detach().cpu().numpy()

    std_per_dim = X_np.std(axis=0)
    print(f"  std mỗi chiều: min={std_per_dim.min():.6f} max={std_per_dim.max():.6f} "
          f"mean={std_per_dim.mean():.6f}")
    sim = _cosine_diversity(X_np)
    print(f"  avg pairwise cosine similarity giữa X_profile của các user: {sim:.4f}")

    n_unique_rows = len(np.unique(X_np.round(decimals=6), axis=0))
    print(f"  Số profile vector THỰC SỰ khác nhau: {n_unique_rows} / {len(X_np)} users")

    ok = sim < 0.9 and std_per_dim.mean() > 1e-3
    return _verdict(
        ok,
        "X_profile có đủ phương sai giữa user -> GP kernel có thể phân biệt được user.",
        "X_profile GẦN NHƯ HẰNG SỐ giữa các user (cosine sim cao / std rất nhỏ) "
        "-> kernel K(x,x') ~ constant cho mọi cặp -> GP prior ép mọi z_u giống nhau "
        "(ĐÂY LÀ NGUYÊN NHÂN GỐC GÂY COLLAPSE, không phụ thuộc encoder/decoder).",
    )


# --------------------------------------------------------------------------- #
# GT 3: s_u thô (trước khi qua encoder) đã đồng nhất giữa user chưa?
#       (loại trừ khả năng lỗi nằm ở bước SVD + mean-pooling)
# --------------------------------------------------------------------------- #
def check_su_raw_diversity(s_u_full: torch.Tensor, nz_mask: torch.Tensor):
    print("\n[GT-3] s_u thô (mean-pooled SVD item embeddings) có đa dạng không?")
    su_np = s_u_full[nz_mask].detach().cpu().numpy()
    sim = _cosine_diversity(su_np)
    norms = np.linalg.norm(su_np, axis=1)
    print(f"  avg pairwise cosine similarity của s_u thô: {sim:.4f}")
    print(f"  norm(s_u): min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f} "
          f"std={norms.std():.4f}")

    ok = sim < 0.85
    return _verdict(
        ok,
        "s_u thô có đủ đa dạng -> input vào encoder không phải nguyên nhân collapse.",
        "s_u thô ĐÃ rất giống nhau giữa user trước khi vào model (cosine sim cao) "
        "-> có thể do popularity bias quá mạnh trong bước SVD/mean-pooling, khiến "
        "personalization signal bị xoá từ gốc, độc lập với GP/encoder.",
    )


# --------------------------------------------------------------------------- #
# GT 4: Encoder có tự nó collapse không? (bỏ GP ra khỏi vòng lặp, test trực tiếp)
#       Đưa input rất khác nhau (kể cả input giả/synthetic), xem mu có khác nhau không.
# --------------------------------------------------------------------------- #
def check_encoder_collapse(model, s_u_full: torch.Tensor, nz_mask: torch.Tensor,
                            device, n_sample=2000, seed=0):
    print("\n[GT-4] Encoder (độc lập với GP) có tự collapse không?")
    model.eval()
    su_np_all = s_u_full[nz_mask]
    n = su_np_all.size(0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    su_real = su_np_all[idx].to(device)

    with torch.no_grad():
        mu_real, _ = model.encode(su_real)
    sim_real = _cosine_diversity(mu_real.cpu().numpy())
    print(f"  [Input = s_u THẬT]      avg cosine sim của mu: {sim_real:.4f}")

    # input synthetic CỰC KỲ khác nhau (random Gaussian std lớn) để xem encoder
    # có khả năng tạo output khác nhau về NGUYÊN TẮC hay không (kiểm tra capacity,
    # không liên quan tới việc s_u thật có đa dạng hay không)
    synth = torch.randn(min(n_sample, n), model.m_dim, device=device) * 5.0
    with torch.no_grad():
        mu_synth, _ = model.encode(synth)
    sim_synth = _cosine_diversity(mu_synth.cpu().numpy())
    print(f"  [Input = Gaussian random std=5] avg cosine sim của mu: {sim_synth:.4f}")

    ok_capacity = sim_synth < 0.5
    ok_real = sim_real < 0.85

    _verdict(
        ok_capacity,
        "Encoder VỀ NGUYÊN TẮC vẫn phân biệt được input khác nhau (không bị dead/saturated).",
        "Encoder collapse NGAY CẢ với input cực kỳ khác nhau (random std=5) "
        "-> bệnh nằm ở encoder weights tự nó (bị regularize quá mạnh / dead ReLU / "
        "learning rate quá cao làm nổ rồi sập về 1 điểm). Không liên quan gì đến GP nữa.",
    )
    if ok_capacity and not ok_real:
        print("  [CHẨN ĐOÁN] Encoder CÓ capacity phân biệt input, nhưng với s_u THẬT lại "
              "collapse -> rất có khả năng do GP-term trong loss học được cách ép encoder "
              "bỏ qua s_u thật (xem GT-2/GT-5).")
    return ok_capacity, ok_real


# --------------------------------------------------------------------------- #
# GT 5: GP kernel matrix tính trên 1 batch train thật trông như thế nào?
#       (ma trận gần như constant => xác nhận trực tiếp cơ chế gây collapse)
# --------------------------------------------------------------------------- #
def check_gp_kernel_structure(model, X_profile_batch: torch.Tensor, device):
    print("\n[GT-5] Ma trận kernel GP trên 1 batch thật có bị suy biến (gần hằng số) không?")
    model.eval()
    X = X_profile_batch.to(device)
    with torch.no_grad():
        K = model.compute_gp_covariance(X)
    K_np = K.cpu().numpy()
    diag = np.diag(K_np)
    off_diag_mask = ~np.eye(K_np.shape[0], dtype=bool)
    off_diag = K_np[off_diag_mask]

    print(f"  sigma_f={model.sigma_f.item():.4f} | alpha={model.alpha.item():.4f}")
    print(f"  K diag: mean={diag.mean():.4f} std={diag.std():.4f}")
    print(f"  K off-diag: mean={off_diag.mean():.4f} std={off_diag.std():.4f} "
          f"min={off_diag.min():.4f} max={off_diag.max():.4f}")
    ratio = off_diag.mean() / (model.sigma_f.item() ** 2 + 1e-8)
    print(f"  Tỉ lệ off-diag / sigma_f^2 (gần 1.0 = kernel suy biến thành hằng số): {ratio:.4f}")

    ok = ratio < 0.8
    return _verdict(
        ok,
        "Kernel off-diagonal đủ thấp/đa dạng so với sigma_f^2 -> GP còn phân biệt user.",
        "Off-diagonal của K GẦN BẰNG sigma_f^2 cho hầu hết cặp user -> kernel suy biến "
        "thành ma trận gần-hằng-số (mọi user 'giống nhau' dưới mắt GP) -> đây CHÍNH LÀ "
        "cơ chế ép z_u (và do đó mu của encoder) hội tụ về 1 điểm chung khi train "
        "(xem giải thích trong cfls_model.py docstring 'kernel collapse').",
    )


# --------------------------------------------------------------------------- #
# GT 6: tau hiện tại so với scale thực tế của X_profile — có "nuốt" hết tín hiệu không?
# --------------------------------------------------------------------------- #
def check_tau_scale_mismatch(model, X_profile_batch: torch.Tensor):
    print("\n[GT-6] tau (length-scale GP) có khớp với scale thực tế của X_profile không?")
    X_np = X_profile_batch.detach().cpu().numpy()
    feat_std = X_np.std(axis=0)
    tau_np = model.tau.detach().cpu().numpy()
    print(f"  std(X_profile) theo từng chiều: mean={feat_std.mean():.5f} max={feat_std.max():.5f}")
    print(f"  tau hiện tại: mean={tau_np.mean():.5f} (khởi tạo mặc định = 1.0)")
    print(f"  Xs = X/tau -> std hiệu dụng: mean={(feat_std/ (tau_np+1e-8)).mean():.5f}")

    ratio = feat_std.mean() / (tau_np.mean() + 1e-8)
    ok = ratio > 0.05
    return _verdict(
        ok,
        "Scale của X/tau đủ lớn để khoảng cách giữa user khác 0 rõ rệt.",
        "std(X)/tau RẤT NHỎ (X_profile có scale nhỏ hơn tau rất nhiều, vd FM embedding "
        "init std=0.01 nhưng tau init =1.0) -> dist_sq giữa MỌI cặp user ~ 0 "
        "-> kernel suy biến thành hằng số ngay từ epoch đầu, trước khi log_tau kịp học "
        "đủ nhanh để bù lại (learning rate của log_tau bị scale x0.1 theo HYPERPARAM_LR_SCALE, "
        "và bị KHÓA gradient trong vài epoch đầu do warm-up).",
    )


# --------------------------------------------------------------------------- #
# GT 7: Model có đang học nghiệm "lười" (luôn đoán ~ vector trung bình) không?
#       So sánh SSE thật của model với SSE của baseline "đoán mean(s_u)".
# --------------------------------------------------------------------------- #
def check_lazy_mean_solution(model, s_u_full: torch.Tensor, nz_mask: torch.Tensor, device):
    print("\n[GT-7] Model có đang học nghiệm 'lười' (decode ~ vector trung bình) không?")
    model.eval()
    su = s_u_full[nz_mask].to(device)
    mean_vec = su.mean(dim=0, keepdim=True)

    baseline_sse = torch.mean((su - mean_vec) ** 2).item()
    with torch.no_grad():
        mu, _ = model.encode(su)
        s_hat = model.decoder(mu)
    model_sse = torch.mean((su - s_hat) ** 2).item()

    print(f"  SSE baseline 'luôn đoán mean(s_u)' : {baseline_sse:.6f}")
    print(f"  SSE thật của model hiện tại        : {model_sse:.6f}")
    ratio = model_sse / (baseline_sse + 1e-12)
    print(f"  Tỉ lệ model_sse / baseline_sse      : {ratio:.4f}  (gần 1.0 = model ~ nghiệm lười)")

    s_hat_np = s_hat.cpu().numpy()
    centered = s_hat_np - s_hat_np.mean(axis=0, keepdims=True)
    sim_centered = _cosine_diversity(centered)
    sim_raw = _cosine_diversity(s_hat_np)
    print(f"  cosine sim s_hat THÔ (chưa trừ mean)   : {sim_raw:.4f}")
    print(f"  cosine sim s_hat ĐÃ TRỪ MEAN            : {sim_centered:.4f}  "
          f"(đây mới là độ đa dạng THẬT, loại trừ phần bias chung)")

    ok = ratio < 0.7
    return _verdict(
        ok,
        "Model giảm SSE đáng kể so với baseline mean -> đang học personalization thật.",
        "Model SSE ~ XẤP XỈ baseline 'đoán mean' -> ĐÂY LÀ NGUYÊN NHÂN GỐC: vì s_u có "
        "norm tuyệt đối rất nhỏ, recon loss không tạo đủ áp lực gradient để model học "
        "cá nhân hoá thật, nên nó sa vào nghiệm lười 'luôn đoán gần đúng vector trung bình', "
        "khiến top-K ranking bị chi phối bởi thành phần chung (≈ popularity) cho mọi user.",
    )


# --------------------------------------------------------------------------- #
def run_all_diagnostics(model, fm_model, weights_path, s_u_full, nz_mask,
                          all_X_profile, train_X_nz, device):
    print("=" * 70)
    print("CHẨN ĐOÁN COLLAPSE — CFLS PIPELINE")
    print("=" * 70)

    r1 = check_fm_weights_loaded(fm_model, weights_path)
    r2 = check_profile_diversity(all_X_profile, user_idxs=None)
    r3 = check_su_raw_diversity(s_u_full, nz_mask)
    r4_cap, r4_real = check_encoder_collapse(model, s_u_full, nz_mask, device)

    # batch nhỏ thật để soi kernel + tau (lấy ngẫu nhiên <=512 user từ train_X_nz)
    n = min(512, train_X_nz.size(0))
    idx = torch.randperm(train_X_nz.size(0))[:n]
    batch_X = train_X_nz[idx]
    r5 = check_gp_kernel_structure(model, batch_X, device)
    r6 = check_tau_scale_mismatch(model, batch_X)
    r7 = check_lazy_mean_solution(model, s_u_full, nz_mask, device)

    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    table = [
        ("GT-1 FM weights load đúng",        r1),
        ("GT-2 X_profile đa dạng giữa user", r2),
        ("GT-3 s_u thô đa dạng",             r3),
        ("GT-4a Encoder có capacity",        r4_cap),
        ("GT-4b Encoder không collapse w/ s_u thật", r4_real),
        ("GT-5 GP kernel không suy biến",    r5),
        ("GT-6 tau khớp scale X_profile",    r6),
        ("GT-7 Model không học nghiệm lười (mean)", r7),
    ]
    for name, ok in table:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")

    n_fail = sum(1 for _, ok in table if not ok)
    if n_fail == 0:
        print("\n-> Không phát hiện nguyên nhân collapse rõ ràng qua các test này. "
              "Cần soi sâu hơn vào hyperparameter / data thực tế.")
    else:
        print(f"\n-> {n_fail}/8 test FAIL. Ưu tiên sửa theo đúng thứ tự GT-1 -> GT-7 ở trên, "
              "vì các nguyên nhân có thể chồng lấp (vd GT-1 fail thường kéo theo GT-2, GT-5, GT-6 fail).")
    return table


if __name__ == "__main__":
    print(__doc__)
    print("Đây là module chẩn đoán, hãy import run_all_diagnostics(...) và gọi trong train.py "
          "ngay sau khi đã có đủ các biến: model, fm_model, s_u_full, nz_mask, all_X_profile, "
          "train_X_nz, device. Ví dụ:\n")
    print("""
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
    """)