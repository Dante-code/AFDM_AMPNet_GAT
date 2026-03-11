"""
GAT-OTFS 初始化模块 —— 对应论文 Section III-B-1 (Initialization Module)

论文公式对应关系：
  - 公式(14): n_i = [y^T h_i,  h_i^T h_i,  σ²]        节点状态信息
  - 公式(15): e_ij = -h_i^T h_j,  j ∈ N(i)             边特征（标量）
  - 公式(25): N(i) = {∪_j r(j) \\ {i, i+MN} | j ∈ c(i)} 邻接列表
  - 公式(26): u_i^0 = Φ(n_i; θ)                         初始节点特征

与 GitHub 现有代码 (AMP-GAT) 的主要区别：
  1. 节点特征为 3 维 [y^T h_i, h_i^T h_i, σ²]，而非 4 维 [y^T h_i, h_i^T h_i, r_i, ν_r_i]
  2. 边特征为标量 e_ij = -h_i^T h_j, 而非 3 维 gram_triplet
  3. 没有 AMP 外层循环, sigma^2 直接作为节点特征输入
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# 1) 邻接矩阵构建  —— 对应论文公式 (16) / (25)
# ============================================================
def build_adjacency(
    H: torch.Tensor,
    eps: float = 1e-8,
    add_self_loop: bool = True,
) -> torch.Tensor:
    """
    根据信道矩阵 H 的 Gram 矩阵构建稀疏邻接矩阵。

    论文判据（公式 16）：
        若 h_i^T h_j ≠ 0，则节点 j 是节点 i 的邻居。
        h_i^T h_j 就是 Gram 矩阵 G = H^T H 的第 (i,j) 元素。

    参数:
        H:    (B, n, n)  实值等效信道矩阵，n = 2MN
        eps:  判定非零的阈值
        add_self_loop: 是否强制添加自环（论文未显式说明，
                       但 GNN 实践中通常需要自环保证数值稳定）

    返回:
        adj:  (B, n, n)  float 掩码，1 表示有边，0 表示无边
    """
    # G[b,i,j] = h_i^T h_j  （列向量的内积）
    G = torch.bmm(H.transpose(1, 2), H)  # (B, n, n)
    adj = (torch.abs(G) > eps).float()

    if add_self_loop:
        B, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
        eye = eye.unsqueeze(0).expand(B, -1, -1)
        adj = torch.maximum(adj, eye)

    return adj


# ============================================================
# 2) 边特征构建  —— 对应论文公式 (15)
# ============================================================
def build_edge_features(H: torch.Tensor) -> torch.Tensor:
    """
    构建论文定义的标量边特征。

    论文公式 (15)：
        e_ij = -h_i^T h_j,  j ∈ N(i)

    与 GitHub 代码的区别：
        GitHub 使用 gram_triplet 模式，返回 (B,n,n,3)。
        论文仅使用单个标量，返回 (B,n,n,1)。

    参数:
        H:  (B, n, n)

    返回:
        edge_feat:  (B, n, n, 1)  标量边特征，最后一维保留方便后续拼接
    """
    G = torch.bmm(H.transpose(1, 2), H)  # (B, n, n)
    edge_feat = -G                        # e_ij = -h_i^T h_j
    return edge_feat.unsqueeze(-1)        # (B, n, n, 1)


# ============================================================
# 3) 节点状态信息提取  —— 对应论文公式 (14)
# ============================================================
def extract_node_status(
    y: torch.Tensor,
    H: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """
    提取每个节点的状态信息向量 n_i。

    论文公式 (14)：
        n_i = [y^T h_i,  h_i^T h_i,  σ²]

    其中：
      - y^T h_i:   接收信号在第 i 个信道列方向上的投影（匹配滤波输出）
      - h_i^T h_i:  第 i 个信道列的能量
      - σ²:        噪声方差（所有节点共享同一标量值）

    与 GitHub 代码的区别：
      - GitHub 使用 [y^T h_i, h_i^T h_i, r_i, ν_r_i]（4 维，含 AMP 先验）
      - 论文使用 [y^T h_i, h_i^T h_i, σ²]（3 维，含噪声方差）

    参数:
        y:      (B, n)     接收信号
        H:      (B, n, n)  信道矩阵
        sigma2: (B,)       噪声方差

    返回:
        n_i:    (B, n, 3)  每个节点的 3 维状态向量
    """
    # y^T h_i:  (B, 1, n) @ (B, n, n) -> (B, 1, n) -> (B, n)
    yh = torch.bmm(y.unsqueeze(1), H).squeeze(1)

    # h_i^T h_i = ||h_i||^2:  对 H 的每一列求平方和
    # H 的形状 (B, n, n)，H[:,j,:] 不是第 j 列；H[:,:,j] 才是第 j 列
    # 但 (H**2).sum(dim=1) 其实等于 diag(H^T H)，这正好是 h_i^T h_i
    hh = (H ** 2).sum(dim=1)  # (B, n)

    # σ²: (B,) -> 广播到 (B, n)
    sigma2_expand = sigma2.unsqueeze(-1).expand_as(yh)  # (B, n)

    # 拼接为 3 维特征
    n_i = torch.stack([yh, hh, sigma2_expand], dim=-1)  # (B, n, 3)

    return n_i


# ============================================================
# 4) 初始节点特征投影  —— 对应论文公式 (26)
# ============================================================
class NodeInitFFN(nn.Module):
    """
    论文公式 (26)：u_i^0 = Φ(n_i; θ)

    Φ 是一个单层前馈网络：
        输入维度 = 3  (即 n_i 的维度)
        输出维度 = F  (论文默认 F = 8)

    论文原文：
        "Φ(·; θ) denotes a single-layer feed-forward neural network
         with an input dimension of 3 and an output dimension of F"
    """

    def __init__(self, F: int = 8):
        """
        参数:
            F: 节点特征维度，论文默认 F = 8
        """
        super().__init__()
        self.F = F
        self.ffn = nn.Linear(3, F)

    def forward(self, n_i: torch.Tensor) -> torch.Tensor:
        """
        参数:
            n_i: (B, n, 3)  来自 extract_node_status 的节点状态信息

        返回:
            u_0: (B, n, F)  初始节点特征
        """
        return self.ffn(n_i)


# ============================================================
# 5) 整合：完整初始化模块
# ============================================================
class GATOTFSInitModule(nn.Module):
    """
    将上述所有组件封装为一个完整的初始化模块。

    输入:  y, H, σ²
    输出:  u_0 (初始节点特征), adj (邻接矩阵), edge_feat (边特征)

    使用示例:
        init_module = GATOTFSInitModule(F=8)
        u_0, adj, edge_feat = init_module(y, H, sigma2)
    """

    def __init__(self, F: int = 8, adj_eps: float = 1e-8, add_self_loop: bool = True):
        super().__init__()
        self.F = F
        self.adj_eps = adj_eps
        self.add_self_loop = add_self_loop
        self.node_init = NodeInitFFN(F=F)

    def forward(
        self,
        y: torch.Tensor,       # (B, n)
        H: torch.Tensor,       # (B, n, n)
        sigma2: torch.Tensor,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            u_0:       (B, n, F)    初始节点特征
            adj:       (B, n, n)    邻接矩阵
            edge_feat: (B, n, n, 1) 标量边特征
        """
        # 1. 邻接矩阵
        adj = build_adjacency(H, eps=self.adj_eps, add_self_loop=self.add_self_loop)

        # 2. 边特征
        edge_feat = build_edge_features(H)

        # 3. 节点状态信息
        n_i = extract_node_status(y, H, sigma2)

        # 4. 初始节点特征投影
        u_0 = self.node_init(n_i)

        return u_0, adj, edge_feat


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟参数: M=4, N=4, QPSK -> n = 2*M*N = 32
    B, M, N = 2, 4, 4
    n = 2 * M * N  # 32

    # 模拟输入
    y = torch.randn(B, n)
    H = torch.randn(B, n, n) * 0.1     # 随机信道（实际应为稀疏矩阵）
    sigma2 = torch.tensor([0.1, 0.2])   # 两个样本不同的噪声方差

    # ---- 测试各子模块 ----

    # (a) 邻接矩阵
    adj = build_adjacency(H)
    print(f"adj shape:       {adj.shape}")        # (2, 32, 32)
    print(f"adj 稀疏度:      {(adj == 0).float().mean():.2%}")

    # (b) 边特征
    edge_feat = build_edge_features(H)
    print(f"edge_feat shape: {edge_feat.shape}")   # (2, 32, 32, 1)

    # (c) 节点状态信息
    n_i = extract_node_status(y, H, sigma2)
    print(f"n_i shape:       {n_i.shape}")         # (2, 32, 3)
    print(f"n_i[0,0]:        {n_i[0, 0]}")         # [y^T h_0, h_0^T h_0, σ²]

    # (d) 初始投影
    init_ffn = NodeInitFFN(F=8)
    u_0 = init_ffn(n_i)
    print(f"u_0 shape:       {u_0.shape}")         # (2, 32, 8)

    # (e) 完整初始化模块
    init_module = GATOTFSInitModule(F=8)
    u_0_full, adj_full, edge_full = init_module(y, H, sigma2)
    print(f"\n--- 完整模块输出 ---")
    print(f"u_0:       {u_0_full.shape}")           # (2, 32, 8)
    print(f"adj:       {adj_full.shape}")            # (2, 32, 32)
    print(f"edge_feat: {edge_full.shape}")           # (2, 32, 32, 1)

    # (f) 验证数值正确性
    # y^T h_i 应等于 (y @ H)_i
    yH_manual = (y.unsqueeze(1) @ H).squeeze(1)
    assert torch.allclose(n_i[:, :, 0], yH_manual), "y^T h_i 计算有误"

    # h_i^T h_i 应等于 diag(H^T H)
    G = H.transpose(1, 2) @ H
    diag_G = torch.diagonal(G, dim1=1, dim2=2)
    assert torch.allclose(n_i[:, :, 1], diag_G), "h_i^T h_i 计算有误"

    # σ² 每个节点应相同
    assert torch.allclose(n_i[0, :, 2], sigma2[0].expand(n)), "σ² 广播有误"

    # 边特征应等于 -G
    assert torch.allclose(edge_feat.squeeze(-1), -G), "e_ij 计算有误"

    print("\n所有数值验证通过！")