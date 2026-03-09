"""
轻量 GNN 模块：Pair-wise MRF + Propagation + Update + Readout
对应论文 thesis.md III.C，暂不做 IDI 近似（AMP-GNN-v2 风格）
QPSK: Q_R = {-1/sqrt(2), 1/sqrt(2)} 实部/虚部各 2 级
"""
import torch
import torch.nn as nn
import numpy as np

# QPSK 实部星座点（单位平均功率）
Q_R = np.array([-1.0/np.sqrt(2), 1.0/np.sqrt(2)], dtype=np.float32)
N_Q = len(Q_R)


def build_adjacency(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    H: (B, 2MN, 2MN)
    返回邻接矩阵 adj: (B, 2MN, 2MN)，adj[b,i,j]=1 若 h_i^T h_j != 0
    h_i^T h_j = (H^T H)[i,j]，即 Gram 矩阵
    """
    G = torch.bmm(H.transpose(1, 2), H)  # (B, n, n)
    adj = (torch.abs(G) > eps).float()
    return adj


class GNNModule(nn.Module):
    """
    轻量 GNN：节点特征 + 消息聚合 + GRU 更新 + Softmax 读出
    """

    def __init__(self, n_dim: int, n_u: int = 8, n_h: int = 12, n_conv: int = 2,
                 n_mlp_hidden: int = 16):
        super().__init__()
        self.n_dim = n_dim
        self.n_u = n_u
        self.n_h = n_h
        self.n_conv = n_conv
        self.register_buffer('Q_R', torch.tensor(Q_R, dtype=torch.float32))

        # 初始特征: [y^T h_i, h_i^T h_i] -> n_u
        self.W1 = nn.Linear(2, n_u)
        self.b1 = nn.Parameter(torch.zeros(n_u))

        # 消息聚合 MLP: 输入 [u_i, u_j, e_ij] 拼接后聚合，维度 3*n_u
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * n_u, n_mlp_hidden),
            nn.ReLU(),
            nn.Linear(n_mlp_hidden, n_u),
        )

        # GRU
        self.gru = nn.GRUCell(n_u + 2, n_h)  # 输入 [m_i, a_i], a_i=[r_i, nu_r_i]

        # 输出层
        self.W2 = nn.Linear(n_h, n_u)
        self.b2 = nn.Parameter(torch.zeros(n_u))

        # Readout MLP: n_u -> N_Q (softmax)
        self.readout = nn.Sequential(
            nn.Linear(n_u, n_mlp_hidden),
            nn.ReLU(),
            nn.Linear(n_mlp_hidden, N_Q),
        )

    def forward(self, y: torch.Tensor, H: torch.Tensor, r: torch.Tensor, nu_r: torch.Tensor,
                adj: torch.Tensor) -> tuple:
        """
        y: (B, n)
        H: (B, n, n)
        r, nu_r: (B, n) 来自 AMP
        adj: (B, n, n) 邻接矩阵
        返回: x_hat (B, n), nu_x (B, n)
        """
        B, n = y.shape
        device = y.device

        # 初始特征 u_i^(0) = W1 * [y^T h_i, h_i^T h_i] + b1
        yh = torch.bmm(y.unsqueeze(1), H).squeeze(1)   # (B, n)
        hh = (H ** 2).sum(dim=1)                        # (B, n) 即 h_i^T h_i
        feat = torch.stack([yh, hh], dim=-1)             # (B, n, 2)
        u = self.W1(feat) + self.b1                     # (B, n, n_u)

        # 边权 e_ij = h_i^T h_j
        G = torch.bmm(H.transpose(1, 2), H)
        a_i = torch.stack([r, nu_r.clamp(min=1e-10)], dim=-1)  # (B, n, 2)

        s = torch.zeros(B, n, self.n_h, device=device)
        for _ in range(self.n_conv):
            # Propagation: m_i = f_theta(sum_{j in N(i)} [u_i, u_j, e_ij])
            u_i = u.unsqueeze(2).expand(-1, -1, n, -1)   # (B, n, n, n_u)
            u_j = u.unsqueeze(1).expand(-1, n, -1, -1)   # (B, n, n, n_u)
            e_ij = G.unsqueeze(-1).expand(-1, -1, -1, self.n_u)
            msg_j2i = torch.cat([u_i, u_j, e_ij], dim=-1)  # (B, n, n, 3*n_u)

            # 只对 adj[b,i,j]=1 的 j 聚合
            msg_j2i = msg_j2i * adj.unsqueeze(-1)
            agg = msg_j2i.sum(dim=2)  # (B, n, 3*n_u)
            m = self.msg_mlp(agg)  # (B, n, n_u)

            # Update: s_i = GRU(s_i, [m_i, a_i]), u_i = W2*s_i + b2
            gru_in = torch.cat([m, a_i], dim=-1)
            s = self.gru(gru_in.reshape(B * n, -1), s.reshape(B * n, -1))
            s = s.reshape(B, n, -1)
            u = self.W2(s) + self.b2

        # Readout: softmax -> p(x_i=s), x_hat = sum s*p, nu_x = sum (s-x_hat)^2 * p
        logits = self.readout(u)  # (B, n, N_Q)
        p = torch.softmax(logits, dim=-1)

        Q = self.Q_R.to(device).view(1, 1, -1)
        x_hat = (p * Q).sum(dim=-1)
        nu_x = (p * (Q - x_hat.unsqueeze(-1)) ** 2).sum(dim=-1)
        nu_x = nu_x.clamp(min=1e-10)

        return x_hat, nu_x


def build_gnn(n_dim: int, **kwargs) -> GNNModule:
    return GNNModule(n_dim, **kwargs)
