"""
AMP 线性模块：仅输出 (r, nu_r)，不做星座去噪
对应论文 thesis.md III.B 公式 (65)-(68)
输入输出均为实值，适用于 PyTorch 训练
"""
import torch
import torch.nn as nn


def amp_linear_step(y: torch.Tensor, H: torch.Tensor, x_hat: torch.Tensor,
                    nu_x: torch.Tensor, z_prev: torch.Tensor, nu_z_prev: torch.Tensor,
                    sigma2: torch.Tensor, eps: float = 1e-10) -> tuple:
    """
    单步 AMP 更新（实值）
    y: (B, 2MN)
    H: (B, 2MN, 2MN)
    x_hat, nu_x: (B, 2MN)
    z_prev, nu_z_prev: (B, 2MN)
    sigma2: (B,) 实值噪声方差 σ_n^2/2
    返回: z, nu_z, r, nu_r
    """
    # sigma2 扩展为 (B, 1) 便于广播
    s2 = sigma2.view(-1, 1) + eps

    # nu_z_j = sum_i h_ji^2 * nu_x_i
    H2 = H ** 2
    nu_z = torch.bmm(H2, nu_x.unsqueeze(-1)).squeeze(-1) + eps

    # z_j = sum_i h_ji * x_hat_i - (nu_z * (y - z_prev)) / (nu_z_prev + sigma2/2)
    Hz = torch.bmm(H, x_hat.unsqueeze(-1)).squeeze(-1)
    denom = nu_z_prev + s2
    z = Hz - (nu_z / denom) * (y - z_prev)

    # nu_r_i = 1 / (sum_j h_ij^2 / (nu_z_j + sigma2/2))
    # H^T 的 (i,j) 即 H 的 (j,i)，所以 sum_j h_ji^2 / (...) 即 H2.T @ (1/(nu_z+s2))
    inv_denom = 1.0 / (nu_z + s2)
    denom_r = torch.bmm(H2.transpose(1, 2), inv_denom.unsqueeze(-1)).squeeze(-1) + eps
    nu_r = 1.0 / denom_r

    # r_i = x_hat_i + nu_r_i * sum_j h_ij * (y_j - z_j) / (nu_z_j + sigma2/2)
    residual = (y - z) * inv_denom
    HTr = torch.bmm(H.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
    r = x_hat + nu_r * HTr

    return z, nu_z, r, nu_r


def amp_linear_forward(y: torch.Tensor, H: torch.Tensor, x_hat_init: torch.Tensor,
                       nu_x_init: torch.Tensor, sigma2: torch.Tensor,
                       n_iter: int = 1, eps: float = 1e-10) -> tuple:
    """
    AMP 线性前向（多步，用于单次 AMP 调用，如第一轮）
    从 x_hat, nu_x 出发，跑 n_iter 步，返回最后的 r, nu_r
    注意：这里每步用相同的 x_hat, nu_x（不更新），因为更新由 GNN 做
    所以实际上 n_iter=1 即可，多步没有意义除非我们做纯 AMP 迭代
    对于 AMP-GNN，每轮 t: AMP 用 (x_hat^(t-1), nu_x^(t-1)) 得到 (r, nu_r)，然后 GNN 输出 (x_hat^t, nu_x^t)
    因此这里只做 1 步 AMP 更新即可。
    """
    B, n = y.shape
    z = y.clone()
    nu_z = torch.bmm((H ** 2), nu_x_init.unsqueeze(-1)).squeeze(-1) + eps
    s2 = sigma2.view(-1, 1) + eps

    for _ in range(n_iter - 1):
        z, nu_z, r, nu_r = amp_linear_step(y, H, x_hat_init, nu_x_init, z, nu_z, sigma2, eps)
        # 纯 AMP 时我们会更新 x_hat, nu_x；这里不更新，所以多步等价

    z, nu_z, r, nu_r = amp_linear_step(y, H, x_hat_init, nu_x_init, z, nu_z, sigma2, eps)
    return r, nu_r
