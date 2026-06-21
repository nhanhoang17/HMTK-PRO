"""
cfls_model.py
-------------
FIX: Clamp sigma_f va alpha tranh kernel collapse.

Dau hieu collapse: sigma_f -> 0, alpha -> lon
-> K(x,x') ~ alpha*I -> GP mat tac dung, moi user doc lap nhau
-> Cold-start prediction = prior mean ~ 0, khong mang thong tin CF

Fix:
  sigma_f >= 0.1  (dam bao kernel van phan biet duoc users)
  alpha   <= 0.5  (dam bao noise khong at len signal)
"""

import math
import torch
import torch.nn as nn


class CFLS(nn.Module):
    def __init__(self, m_dim, k_dim, profile_dim, dropout=0.1):
        super().__init__()
        self.m_dim = m_dim
        self.k_dim = k_dim

        # Encoder: 2-layer MLP + ReLU (paper Sect. 4.2)
        self.encoder = nn.Sequential(
            nn.Linear(m_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),  nn.ReLU(), nn.Dropout(dropout),
        )
        self.fc_mu     = nn.Linear(256, k_dim)
        self.fc_logvar = nn.Linear(256, k_dim)

        # Decoder: 2-layer MLP + ReLU (paper Sect. 4.2)
        self.decoder = nn.Sequential(
            nn.Linear(k_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, m_dim),
        )

        # GP kernel parameters (Eq. 5)
        # Khoi tao sigma_f = 1.0, alpha = 0.1
        self.log_sigma_f = nn.Parameter(torch.zeros(1))           # exp(0) = 1.0
        self.log_tau     = nn.Parameter(torch.zeros(profile_dim))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(0.1)))

        # sigma_s_sq = 1.0: hang so, khong hoc
        self.register_buffer("sigma_s_sq", torch.tensor(1.0))

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