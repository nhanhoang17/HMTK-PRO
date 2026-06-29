"""
embeddings.py
-------------
EmbeddingEngine : SVD song embeddings + su vectors (sparse matmul, tranh OOM)
FactorizationMachine : hoc dense user profile xu
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import svds


class EmbeddingEngine:

    @staticmethod
    def get_svd_item_embeddings(sparse_matrix, k_dim):
        """
        v_i = Vt.T * sqrt(sigma) in R^M  (Sect. 3.1)
        Input: log1p sparse matrix tu build_interaction_matrix.
        """
        print(f"Chay SVD ({k_dim} chieu) ...")
        mat = sparse_matrix.copy().astype(np.float32)
        U, sigma, Vt = svds(mat, k=k_dim)
        sigma = sigma[::-1]; Vt = Vt[::-1, :]
        item_emb = (Vt.T) * np.sqrt(sigma)   # (n_items, k_dim)
        print(f"  Song embedding: {item_emb.shape}")
        return torch.tensor(item_emb, dtype=torch.float32)

    @staticmethod
    def prepare_su_vectors(sparse_matrix, item_embeddings):
        """
        s_u = (1/|I_u|) * sum_{j in I_u} v_j  (Eq. 3.1)

        Dung sparse matmul thay vi toarray() (tranh OOM 178 GB):
            S = A_bin @ emb / count
        RAM peak: ~emb_size + S_size ~ 120 MB, khong phu thuoc n_users x n_items.
        Toc do: ~0.3s cho 66k users (9x nhanh hon getrow loop).
        """
        print("Tinh s_u vectors (sparse matmul) ...")
        emb_np  = item_embeddings.numpy()
        mat_bin = sparse_matrix.tocsr().copy()
        mat_bin.data = np.ones_like(mat_bin.data)   # binary
        counts  = np.array(mat_bin.sum(axis=1)).flatten()
        counts  = np.where(counts == 0, 1.0, counts)
        S = (mat_bin @ emb_np) / counts[:, None]
        print(f"  s_u: {S.shape}  non-zero: {(S.any(axis=1)).sum()}")
        return torch.tensor(S.astype(np.float32))


class FactorizationMachine(nn.Module):
    """
    FM theo paper (Rendle 2010, ref [17] trong CFLS).
 
    Input: [tz, lang, tlang, track_idx] — cả profile features lẫn item.
    Target: implicit interaction score (log1p count).
    Mục đích: học dense embedding x̃^l cho từng profile feature l,
              sao cho embedding phản ánh cả demographic lẫn sở thích âm nhạc.
 
    Sau khi train, chỉ dùng emb_tz/emb_lang/emb_tlang để rút x_u:
        x_u = concat(emb_tz(u), emb_lang(u), emb_tlang(u)) ∈ R^{3*embed_dim}
    item embedding bị bỏ đi — chỉ là "supervision signal" khi train.
 
    FM score (Eq. gốc Rendle):
        ŷ = b + Σ_i w_i·x_i + Σ_{i<j} <v_i, v_j>·x_i·x_j
    Với one-hot input (mỗi field chỉ có 1 active feature):
        ŷ = b_global + b_tz + b_lang + b_tlang + b_item
            + <v_tz, v_lang> + <v_tz, v_tlang> + <v_tz, v_item>
            + <v_lang, v_tlang> + <v_lang, v_item> + <v_tlang, v_item>
    Interaction term tính hiệu quả bằng: ½(||Σv||² - Σ||v||²)
    """
    def __init__(self, num_tz, num_lang, num_tlang, num_items, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
 
        # Feature embeddings (v_i trong Rendle FM)
        self.emb_tz    = nn.Embedding(num_tz,    embed_dim)
        self.emb_lang  = nn.Embedding(num_lang,  embed_dim)
        self.emb_tlang = nn.Embedding(num_tlang, embed_dim)
        self.emb_item  = nn.Embedding(num_items, embed_dim)
 
        # Linear bias terms (w_i trong Rendle FM)
        self.bias_tz    = nn.Embedding(num_tz,    1)
        self.bias_lang  = nn.Embedding(num_lang,  1)
        self.bias_tlang = nn.Embedding(num_tlang, 1)
        self.bias_item  = nn.Embedding(num_items, 1)
        self.bias_global = nn.Parameter(torch.zeros(1))
 
        # Khởi tạo: embedding nhỏ, bias = 0
        for emb in [self.emb_tz, self.emb_lang, self.emb_tlang, self.emb_item]:
            nn.init.normal_(emb.weight, std=0.01)
        for b in [self.bias_tz, self.bias_lang, self.bias_tlang, self.bias_item]:
            nn.init.zeros_(b.weight)
 
    def forward(self, tz, lang, tlang, item):
        """
        tz, lang, tlang, item: LongTensor (B,)
        Returns: scalar score (B, 1)
        """
        # Embeddings của 4 fields
        e = [self.emb_tz(tz), self.emb_lang(lang),
             self.emb_tlang(tlang), self.emb_item(item)]   # 4 × (B, d)
 
        # Linear terms
        b1 = (self.bias_tz(tz) + self.bias_lang(lang)
              + self.bias_tlang(tlang) + self.bias_item(item))  # (B, 1)
 
        # Interaction term: ½(||Σe||² - Σ||e||²)  — Rendle FM trick
        stk     = torch.stack(e, dim=1)          # (B, 4, d)
        sum_sq  = stk.sum(dim=1) ** 2            # (B, d)
        sq_sum  = (stk ** 2).sum(dim=1)          # (B, d)
        b2 = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)  # (B, 1)
 
        return self.bias_global + b1 + b2        # (B, 1)
 
    @torch.no_grad()
    def extract_user_profiles(self, df_users):
        """
        x_u = concat(emb_tz, emb_lang, emb_tlang) ∈ R^{3*embed_dim}.
        Item embedding bị loại bỏ — chỉ dùng profile part cho GP kernel.
        """
        self.eval()
        tz    = torch.tensor(df_users["tz_idx"].values,    dtype=torch.long)
        lang  = torch.tensor(df_users["lang_idx"].values,  dtype=torch.long)
        tlang = torch.tensor(df_users["tlang_idx"].values, dtype=torch.long)
        return torch.cat([self.emb_tz(tz), self.emb_lang(lang),
                          self.emb_tlang(tlang)], dim=1)  # (N, 3*d)


#------------------------------------------------------------------------------------------------------
