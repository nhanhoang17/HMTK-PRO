import math
import torch
import torch.nn as nn


class CFLS(nn.Module):
    def __init__(self, m_dim, k_dim, profile_dim, dropout=0.3):
        super().__init__()
        self.m_dim = m_dim
        self.k_dim = k_dim

        # Encoder: 2-layer MLP + ReLU (paper Sect. 4.2)
        self.encoder = nn.Sequential(
            nn.Linear(m_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            #nn.Linear(256, 256),  nn.ReLU(), nn.Dropout(dropout),
        )
        self.fc_mu     = nn.Linear(256, k_dim)
        self.fc_logvar = nn.Linear(256, k_dim)

        # Decoder: 2-layer MLP + ReLU (paper Sect. 4.2)
        self.decoder = nn.Sequential(
            nn.Linear(k_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            #nn.Linear(256, 256),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, m_dim),
        )

        # GP kernel parameters (Eq. 5)
        # Khoi tao sigma_f = 1.0, alpha = 0.1
        self.log_sigma_f = nn.Parameter(torch.zeros(1))           # exp(0) = 1.0
        #self.log_tau     = nn.Parameter(torch.zeros(profile_dim))
        #self.log_tau = nn.Parameter(torch.ones(profile_dim) * math.log(0.05))
        self.log_tau = nn.Parameter(torch.ones(profile_dim) * math.log(8.0))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(0.1)))

        # sigma_s_sq (Eq. 2, 9): nay la PARAMETER hoc duoc, khong con
        # la buffer co dinh. Log-parameterization: sigma_s^2 = exp(log_sigma_s_sq)
        # Khoi tao = 1.0 (log(1.0) = 0), giong cach khoi tao sigma_f.
        self.log_sigma_s_sq = nn.Parameter(torch.zeros(1))

    @property
    def sigma_f(self):
        # Clamp min=0.1: dam bao kernel van co signal
        return self.log_sigma_f.exp().clamp(min=0.1)

    @property
    def tau(self):
        return self.log_tau.exp()

    @property
    def alpha(self):
        # Clamp max=0.5: tranh noise at len signal cua kernel
        return self.log_alpha.exp().clamp(max=0.5)

    @property
    def sigma_s_sq(self):
        # Clamp [0.2, 10.0]: tranh hai dau collapse cua reconstruction
        # variance. San duoc nang tu 0.05 -> 0.2 vi quan sat thuc te
        # (batch_size=128, M=128) cho thay san 0.05 bi cham lien tuc
        # tu khoang epoch 12, dau hieu cua suy bien Gaussian-likelihood
        # khi model du capacity de gan nhu "ghi nho" s_u (sse term qua
        # nho, gradient day log(sigma_s^2) ve -inf khong gap can).
        # Khoang gia tri nay van can tinh chinh theo thang gia tri thuc
        # te cua s_u trong dataset cua ban (xem them: chuan hoa s_u
        # truoc khi vao encoder, trong embeddings.py/data_prep.py).
        return self.log_sigma_s_sq.exp().clamp(min=0.2, max=10.0)

    def encode(self, s_u):
        h      = self.encoder(s_u)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-4.0, 4.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, s_u):
        mu, logvar = self.encode(s_u)
        z_u        = self.reparameterize(mu, logvar)
        s_hat      = self.decoder(z_u)
        return s_hat, mu, logvar, z_u

    def compute_gp_covariance(self, X):
        """SE kernel + alpha*I"""
        Xs      = X / self.tau.unsqueeze(0)
        sq      = (Xs**2).sum(dim=1, keepdim=True)
        dist_sq = (sq + sq.T - 2 * Xs @ Xs.T).clamp(min=0)
        K = self.sigma_f**2 * torch.exp(-0.5 * dist_sq)
        K = K + self.alpha * torch.eye(X.size(0), device=X.device)
        return K

    def compute_cross_covariance(self, X_star, X_train):
        Xs      = X_star  / self.tau.unsqueeze(0)
        Xt      = X_train / self.tau.unsqueeze(0)
        sq1     = (Xs**2).sum(dim=1, keepdim=True)
        sq2     = (Xt**2).sum(dim=1, keepdim=True)
        dist_sq = (sq1 + sq2.T - 2 * Xs @ Xt.T).clamp(min=0)
        return self.sigma_f**2 * torch.exp(-0.5 * dist_sq)

    def set_hyperparam_grad(self, flag: bool):
        """
        Bat/tat gradient cho toan bo nhom tham so "hyperparameter"
        (log_sigma_f, log_alpha, log_tau, log_sigma_s_sq) cung luc.
        Dung de warm-up: trong vai epoch dau, giu ca GP kernel params
        VA sigma_s_sq co dinh, chi cho encoder/decoder hoc truoc, tranh
        ca hai phia (GP va reconstruction variance) cung suy bien tu
        luc model con chua hoc duoc bieu dien gi huu ich.
        """
        self.log_sigma_f.requires_grad_(flag)
        self.log_alpha.requires_grad_(flag)
        self.log_tau.requires_grad_(flag)
        self.log_sigma_s_sq.requires_grad_(flag)