from __future__ import annotations

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1,
                 logvar_min=-6.0, logvar_max=2.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for idx in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[idx], dims[idx + 1]),
                nn.LayerNorm(dims[idx + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], hidden_dim))
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), self.logvar_min, self.logvar_max)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        dims = [latent_dim] + [hidden_dim] * num_layers
        for idx in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[idx], dims[idx + 1]),
                nn.LayerNorm(dims[idx + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1,
                 beta=1.0, logvar_min=-6.0, logvar_max=2.0):
        super().__init__()
        self.encoder = VAEEncoder(in_dim, hidden_dim, latent_dim, num_layers, dropout,
                                  logvar_min, logvar_max)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, in_dim, num_layers, dropout)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(logvar * 0.5)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z
