"""
IDI (Inter-Doppler Interference) approximation helpers for AMP-GNN.
"""
from __future__ import annotations

import torch


def compute_idi_stats(
    H: torch.Tensor,
    mask_idi: torch.Tensor,
    x_hat: torch.Tensor,
    nu_x: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Gaussian-approximated IDI mean/variance:
      mu_zeta = H_idi @ x_hat
      sigma2_zeta = (H_idi^2) @ nu_x + sigma2
    """
    H_idi = H * mask_idi.float()
    mu_zeta = torch.bmm(H_idi, x_hat.unsqueeze(-1)).squeeze(-1)
    s2 = sigma2.view(-1, 1) + eps
    sigma2_zeta = torch.bmm(H_idi ** 2, nu_x.unsqueeze(-1)).squeeze(-1) + s2
    sigma2_zeta = sigma2_zeta.clamp(min=eps)
    return mu_zeta, sigma2_zeta


def normalize_signal_and_channel(
    y: torch.Tensor,
    H: torch.Tensor,
    mask_idi: torch.Tensor,
    mu_zeta: torch.Tensor,
    sigma2_zeta: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize observations/channel using IDI stats.
    """
    sigma_zeta = torch.sqrt(sigma2_zeta).clamp(min=eps)
    y_tilde = (y - mu_zeta) / sigma_zeta
    mask_main = ~mask_idi
    H_main = H * mask_main.float()
    H_tilde = H_main / sigma_zeta.unsqueeze(-1)
    return y_tilde, H_tilde

