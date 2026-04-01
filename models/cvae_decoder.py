"""
Heteroscedastic decoder for conditional distribution prediction (paper §4.5 / H3).

p(y | c) via direct (mu, sigma) parameterization:
  mu      = E[y | c]          (mean prediction)
  sigma   = exp(log_sigma)    (per-step aleatoric uncertainty)

Replaces CVAE encoder/KL design with a simpler heteroscedastic head.
Class name and all public method signatures kept for backward compatibility.
"""
import torch
import torch.nn as nn


class CVAEDecoder(nn.Module):
    def __init__(self, context_dim: int, output_dim: int,
                 latent_dim: int = 32, hidden_dim: int = 256,
                 dropout_p: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        # NOTE: latent_dim kept in signature for backward compat but unused

        # Shared backbone: context → hidden representation
        self.backbone = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
        )

        # Mean head: predicts expected return trajectory
        self.mu_head = nn.Linear(hidden_dim, output_dim)

        # Log-sigma head: predicts per-step uncertainty
        self.log_sigma_head = nn.Linear(hidden_dim, output_dim)
        # Initialize bias so sigma starts at ~0.007 (reasonable for hourly returns)
        nn.init.constant_(self.log_sigma_head.bias, -5.0)

        # Confidence head (unchanged from before)
        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def _decode(self, c: torch.Tensor):
        """Returns (mu, log_sigma) from context."""
        h = self.backbone(c)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-10, 2)
        return mu, log_sigma

    def forward(self, context: torch.Tensor, y=None):
        """
        Returns (mu, log_sigma) always.
        y parameter kept for backward compat but ignored.
        """
        return self._decode(context)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Returns (B, n_samples, output_dim) from N(mu, sigma^2)."""
        B = context.size(0)
        c_rep = context.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        mu, log_sigma = self._decode(c_rep)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        samples = mu + sigma * eps
        return samples.view(B, n_samples, self.output_dim)

    @torch.no_grad()
    def predict_interval(self, context: torch.Tensor,
                         n_samples: int = 100, alpha: float = 0.80):
        """Returns (mu, pi_low, pi_high) directly from sigma."""
        mu, log_sigma = self._decode(context)
        sigma = torch.exp(log_sigma)
        # z_alpha for 80% PI = 1.2816
        z_alpha = 1.2816
        pi_low = mu - z_alpha * sigma
        pi_high = mu + z_alpha * sigma
        return mu, pi_low, pi_high


def build_cvae_decoder(context_dim: int, output_dim: int,
                       latent_dim: int = 32, hidden_dim: int = 256,
                       dropout_p: float = 0.1) -> CVAEDecoder:
    return CVAEDecoder(context_dim=context_dim, output_dim=output_dim,
                       latent_dim=latent_dim, hidden_dim=hidden_dim,
                       dropout_p=dropout_p)
