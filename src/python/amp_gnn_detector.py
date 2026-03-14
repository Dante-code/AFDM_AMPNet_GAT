"""
AMP-GNN 检测器：AMP 与 GNN 交替迭代
对应论文 thesis.md Fig.1
"""
import torch
import torch.nn as nn
import amp_linear
import gnn_module
import numpy as np


def real_to_complex_hard(x_hat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """
    x_hat: (B, 2MN) 实值
    返回: (B, MN) 复数，硬判决到 QPSK
    """
    B = x_hat.shape[0]
    MN = M * N
    re = x_hat[:, :MN]
    im = x_hat[:, MN:]
    q = 1.0 / np.sqrt(2)
    qpsk = torch.tensor([q*(1+1j), q*(1-1j), q*(-1+1j), q*(-1-1j)], device=x_hat.device)
    re_q = torch.tensor([-q, q], device=x_hat.device)
    re_idx = (re > 0).long()
    im_idx = (im > 0).long()
    re_dec = re_q[re_idx]
    im_dec = re_q[im_idx]
    return torch.complex(re_dec, im_dec)


class AMPGNNDetector(nn.Module):
    """
    AMP-GNN 检测器
    """

    def __init__(self, n_dim: int, n_iter: int = 3, **gnn_kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.n_iter = n_iter
        self.gnn = gnn_module.GNNModule(n_dim, **gnn_kwargs)

    def forward(self, y: torch.Tensor, H: torch.Tensor, sigma2: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        y: (B, 2MN)
        H: (B, 2MN, 2MN)
        sigma2: (B,)
        adj: (B, 2MN, 2MN)
        返回: x_hat (B, 2MN) 实值估计
        """
        B, n = y.shape
        device = y.device
        eps = 1e-10

        # 初始化
        x_hat = torch.zeros(B, n, device=device)
        nu_x = torch.full((B, n), 1.0, device=device)
        z = y.clone()
        nu_z = torch.bmm((H ** 2), nu_x.unsqueeze(-1)).squeeze(-1) + eps

        for t in range(self.n_iter):
            # AMP 步
            z, nu_z, r, nu_r = amp_linear.amp_linear_step(
                y, H, x_hat, nu_x, z, nu_z, sigma2, eps
            )
            # GNN 步
            x_hat, nu_x = self.gnn(y, H, r, nu_r, adj)

        return x_hat


def compute_ber(x_true: torch.Tensor, x_hat: torch.Tensor, M: int, N: int) -> float:
    """
    x_true: (B, 2MN) 实值真值
    x_hat: (B, 2MN) 实值估计
    QPSK 硬判决后比较
    """
    q = 1.0 / np.sqrt(2)
    # 硬判决
    re_true = (x_true[:, :M*N] > 0).long()
    im_true = (x_true[:, M*N:] > 0).long()
    re_hat = (x_hat[:, :M*N] > 0).long()
    im_hat = (x_hat[:, M*N:] > 0).long()
    err = (re_true != re_hat).sum().item() + (im_true != im_hat).sum().item()
    total = 2 * M * N * x_true.shape[0]
    return err / total


def compute_l2_loss(x_true: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """论文 L2 loss"""
    return ((x_true - x_hat) ** 2).mean()
