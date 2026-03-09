"""
Graph feature helpers for AMP-GNN / AMP-GAT.
"""
from __future__ import annotations

import torch


def build_adjacency(H: torch.Tensor, eps: float = 1e-8, add_self_loop: bool = True) -> torch.Tensor:
    """
    Build adjacency from Gram matrix G = H^T H.
    H: (B, n, n)
    return: adj (B, n, n), float mask in {0,1}
    """
    G = torch.bmm(H.transpose(1, 2), H)
    adj = (torch.abs(G) > eps).float()
    if add_self_loop:
        bsz, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(bsz, -1, -1)
        adj = torch.maximum(adj, eye)
    return adj


def build_edge_attr(H: torch.Tensor, mode: str = "gram_triplet") -> torch.Tensor:
    """
    Build edge features from channel matrix.
    H: (B, n, n)
    mode:
      - "gram": [h_i^T h_j], d_e=1
      - "gram_triplet": [h_i^T h_j, ||h_i||^2, ||h_j||^2], d_e=3
    return: edge_attr (B, n, n, d_e)
    """
    G = torch.bmm(H.transpose(1, 2), H)  # (B,n,n)

    if mode == "gram":
        return G.unsqueeze(-1)

    if mode == "gram_triplet":
        diag = torch.diagonal(G, dim1=1, dim2=2)  # (B,n)
        hi2 = diag.unsqueeze(2).expand_as(G)
        hj2 = diag.unsqueeze(1).expand_as(G)
        return torch.stack([G, hi2, hj2], dim=-1)

    raise ValueError(f"Unsupported edge_attr mode: {mode}")

