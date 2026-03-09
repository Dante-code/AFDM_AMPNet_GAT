"""
AMP-GAT detector: AMP outer loop + GAT inner loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import amp_linear
import gat_module


class AMPGATDetector(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_iter: int = 4,
        damp: float = 0.7,
        n_u: int = 12,
        n_h: int = 12,
        n_conv: int = 2,
        n_heads: int = 2,
        attn_dropout: float = 0.0,
        use_edge_attr: bool = True,
        edge_attr_dim: int = 3,
        n_mlp_hidden: int = 16,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_iter = n_iter
        self.damp = float(damp)
        self.gat = gat_module.GATModule(
            n_dim=n_dim,
            n_u=n_u,
            n_h=n_h,
            n_conv=n_conv,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            use_edge_attr=use_edge_attr,
            edge_attr_dim=edge_attr_dim,
            n_mlp_hidden=n_mlp_hidden,
        )

    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        sigma2: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        y: (B,n), H: (B,n,n), sigma2: (B,), adj: (B,n,n)
        return x_hat: (B,n)
        """
        bsz, n = y.shape
        eps = 1e-10
        device = y.device

        x_hat = torch.zeros(bsz, n, device=device)
        nu_x = torch.full((bsz, n), 0.5, device=device)
        z = y.clone()
        nu_z = torch.bmm((H ** 2), nu_x.unsqueeze(-1)).squeeze(-1) + eps

        for _ in range(self.n_iter):
            z, nu_z, r, nu_r = amp_linear.amp_linear_step(y, H, x_hat, nu_x, z, nu_z, sigma2, eps)
            x_new, nu_new = self.gat(y, H, r, nu_r, adj, edge_attr)

            if self.damp < 1.0:
                x_hat = self.damp * x_new + (1.0 - self.damp) * x_hat
                nu_x = self.damp * nu_new + (1.0 - self.damp) * nu_x
            else:
                x_hat, nu_x = x_new, nu_new
            nu_x = nu_x.clamp(min=eps)

        return x_hat

