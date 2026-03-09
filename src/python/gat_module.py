"""
AMP-GAT graph module.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Q_R = np.array([-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.float32)
N_Q = len(Q_R)


class GATModule(nn.Module):
    """
    GAT-based inference module for AMP-GAT.
    """

    def __init__(
        self,
        n_dim: int,
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
        self.n_u = n_u
        self.n_h = n_h
        self.n_conv = n_conv
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim
        self.register_buffer("Q_R", torch.tensor(Q_R, dtype=torch.float32))

        # [y^T h_i, h_i^T h_i, r_i, nu_r_i] -> u_i
        self.init_proj = nn.Linear(4, n_u)

        # Multi-head projections
        self.W_q = nn.ModuleList([nn.Linear(n_u, n_u, bias=False) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(n_u, n_u, bias=False) for _ in range(n_heads)])
        self.W_v = nn.ModuleList([nn.Linear(n_u, n_u, bias=False) for _ in range(n_heads)])

        if use_edge_attr:
            self.W_e = nn.ModuleList([nn.Linear(edge_attr_dim, n_u, bias=False) for _ in range(n_heads)])
            attn_in_dim = 3 * n_u
        else:
            self.W_e = None
            attn_in_dim = 2 * n_u

        self.attn_vec = nn.ModuleList([nn.Linear(attn_in_dim, 1, bias=False) for _ in range(n_heads)])
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Update + readout
        self.gru = nn.GRUCell(n_u + 2, n_h)
        self.out_proj = nn.Linear(n_h, n_u)
        self.readout = nn.Sequential(
            nn.Linear(n_u, n_mlp_hidden),
            nn.ReLU(),
            nn.Linear(n_mlp_hidden, N_Q),
        )

    def _masked_softmax(self, score: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        score: (B,n,n), adj: (B,n,n)
        """
        mask = adj > 0
        neg_inf = torch.tensor(-1e9, device=score.device, dtype=score.dtype)
        s = torch.where(mask, score, neg_inf)
        alpha = torch.softmax(s, dim=-1)
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
        return alpha

    def _compute_attn_score(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidx: int,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Memory-efficient attention score computation without constructing
        [q_i, k_j, e_ij] huge concatenation tensors.
        """
        attn = self.attn_vec[hidx]
        w = attn.weight[0]  # (2*n_u) or (3*n_u)
        bias = attn.bias[0] if attn.bias is not None else 0.0

        w_q = w[: self.n_u]
        w_k = w[self.n_u : 2 * self.n_u]
        q_term = torch.einsum("bnd,d->bn", q, w_q).unsqueeze(2)  # (B,n,1)
        k_term = torch.einsum("bnd,d->bn", k, w_k).unsqueeze(1)  # (B,1,n)
        score = q_term + k_term + bias

        if self.use_edge_attr and edge_attr is not None:
            w_e = w[2 * self.n_u :]
            # Equivalent to: e = W_e(edge_attr); edge_score = <e, w_e>
            # but avoids materializing e with shape (B,n,n,n_u).
            edge_kernel = torch.matmul(self.W_e[hidx].weight.t(), w_e)  # (d_e,)
            edge_term = torch.einsum("bijd,d->bij", edge_attr, edge_kernel)  # (B,n,n)
            score = score + edge_term

        return self.leaky_relu(score)

    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        r: torch.Tensor,
        nu_r: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        y: (B,n), H: (B,n,n), r/nu_r: (B,n), adj: (B,n,n), edge_attr: (B,n,n,d_e)
        return: x_hat, nu_x (B,n)
        """
        bsz, n = y.shape
        device = y.device

        yh = torch.bmm(y.unsqueeze(1), H).squeeze(1)  # (B,n)
        hh = (H ** 2).sum(dim=1)  # (B,n)
        feat = torch.stack([yh, hh, r, nu_r.clamp(min=1e-10)], dim=-1)  # (B,n,4)
        u = self.init_proj(feat)  # (B,n,n_u)

        state = torch.zeros(bsz, n, self.n_h, device=device, dtype=u.dtype)
        a_i = torch.stack([r, nu_r.clamp(min=1e-10)], dim=-1)  # (B,n,2)

        for _ in range(self.n_conv):
            head_msgs = []
            for hidx in range(self.n_heads):
                q = self.W_q[hidx](u)  # (B,n,n_u)
                k = self.W_k[hidx](u)
                v = self.W_v[hidx](u)

                score = self._compute_attn_score(q, k, hidx, edge_attr=edge_attr)  # (B,n,n)
                alpha = self._masked_softmax(score, adj)
                alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

                msg = torch.bmm(alpha, v)  # (B,n,n_u)
                head_msgs.append(msg)

            m = torch.stack(head_msgs, dim=0).mean(dim=0)  # (B,n,n_u)
            gru_in = torch.cat([m, a_i], dim=-1)
            state = self.gru(gru_in.reshape(bsz * n, -1), state.reshape(bsz * n, -1)).reshape(bsz, n, -1)
            u = self.out_proj(state)

        logits = self.readout(u)  # (B,n,N_Q)
        p = torch.softmax(logits, dim=-1)
        Q = self.Q_R.to(device).view(1, 1, -1)
        x_hat = (p * Q).sum(dim=-1)
        nu_x = (p * (Q - x_hat.unsqueeze(-1)) ** 2).sum(dim=-1).clamp(min=1e-10)
        return x_hat, nu_x
