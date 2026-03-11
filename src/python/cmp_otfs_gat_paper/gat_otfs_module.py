"""
GAT-OTFS 核心网络模块 -- 对应论文 Section III-B-2 (Iterative Updating) + III-B-3 (Readout)

论文公式对应关系:
  - 公式(26): u_0_i  = Phi(n_i; theta)                           初始节点特征 (在 gat_otfs_init.py)
  - 公式(27): alpha_ij = softmax_j(LeakyReLU(a^T [W u_i || W u_j]))  注意力系数
  - 公式(28): m_ji = Psi(u_i, u_j, e_ij, sigma2; beta)          消息 MLP
  - 公式(29): u_t_i = GRU(u_{t-1}_i, sum_j alpha_ij * m_ji)     GRU 节点更新
  - 公式(30): r_i = MLP2(u_T_i)                                  读出映射
  - 公式(31): p_hat(x_i=s_k) = softmax_k(r_ik)                  符号概率
  - 公式(32): x_hat_i = argmax ...                               联合硬判决
  - 公式(33): Loss = -sum p(x_i) log p_hat(x_i)                  交叉熵损失

论文默认超参:
  F  = 8   (节点特征维度)
  F' = 16  (注意力升维后维度)
  T  = 10  (迭代层数)
  Nh1 = 64, Nh2 = 32  (MLP 隐层)
  |S| = 2  (QPSK 实部星座点数)

与 GitHub 现有 AMP-GAT 代码的关键区别:
  1. 单头注意力, 共享 W 升维, 不分 Q/K/V
  2. 消息由 3 层 MLP 生成 (非简单 W_v 投影)
  3. 边特征只进消息 MLP, 不进注意力打分
  4. GRU 输入是聚合消息 (不拼接 AMP 先验)
  5. 读出为 3 层 MLP + argmax 硬判决
  6. 无 AMP 外层循环
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# QPSK 实部星座点 (单位平均功率)
# ---------------------------------------------------------------------------
Q_R = np.array([-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.float32)
N_Q = len(Q_R)  # |S| = 2


class GATOTFSModule(nn.Module):
    """
    GAT-OTFS 核心网络: 迭代更新模块 + 读出模块.

    整体流程:
        u_0 (来自初始化模块)
          -> T 层迭代: [注意力 -> 消息MLP -> 加权聚合 -> GRU更新]
          -> 读出MLP -> softmax -> 概率/硬判决

    参数:
        F:    节点特征维度 (论文默认 8)
        F_prime: 注意力升维后维度 (论文默认 16, 即论文中的 F')
        T:    迭代层数 (论文默认 10)
        Nh1:  MLP 第一隐层维度 (论文默认 64)
        Nh2:  MLP 第二隐层维度 (论文默认 32)
        S:    星座点实部集合大小 (QPSK: 2)
        leaky_slope: LeakyReLU 负半轴斜率 (论文未指定, GAT 惯例 0.2)
    """

    def __init__(
        self,
        F: int = 8,
        F_prime: int = 16,
        T: int = 10,
        Nh1: int = 64,
        Nh2: int = 32,
        S: int = N_Q,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        self.F = F
        self.F_prime = F_prime
        self.T = T
        self.S = S

        # ---- 注意力参数 (公式 27) ----
        # 共享线性变换 W: F -> F'
        self.W_attn = nn.Linear(F, F_prime, bias=False)
        # 注意力向量 a: 2F' -> 1 (拆成 a_left, a_right 用于高效计算)
        self.a_left = nn.Parameter(torch.empty(F_prime))
        self.a_right = nn.Parameter(torch.empty(F_prime))
        nn.init.xavier_uniform_(self.a_left.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_right.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(leaky_slope)

        # ---- 消息 MLP (公式 28) ----
        # 输入: [u_i, u_j, e_ij, sigma2] 维度 = F + F + 1 + 1 = 2F + 2
        # 3 层 MLP: (2F+2) -> Nh1 -> Nh2 -> F
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * F + 2, Nh1),
            nn.ReLU(),
            nn.Linear(Nh1, Nh2),
            nn.ReLU(),
            nn.Linear(Nh2, F),
        )

        # ---- GRU 节点更新 (公式 29) ----
        # input = 聚合后的消息 (维度 F)
        # hidden = u_{t-1} (维度 F)
        self.gru = nn.GRUCell(input_size=F, hidden_size=F)

        # ---- 读出 MLP (公式 30) ----
        # 3 层 MLP: F -> Nh1 -> Nh2 -> |S|
        self.readout_mlp = nn.Sequential(
            nn.Linear(F, Nh1),
            nn.ReLU(),
            nn.Linear(Nh1, Nh2),
            nn.ReLU(),
            nn.Linear(Nh2, S),
        )

        # 星座点集合 (用于软估计, 可选)
        self.register_buffer("Q_R", torch.tensor(Q_R, dtype=torch.float32))

    # =================================================================
    # 注意力系数计算 -- 公式 (27)
    # =================================================================
    def compute_attention(
        self,
        u: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        论文公式 (27):
            beta_ij = LeakyReLU(a^T [W u_i || W u_j])
            alpha_ij = softmax_j(beta_ij),  j in N(i)

        高效实现: 将 a 拆为 [a_left, a_right], 则
            beta_ij = LeakyReLU( (Wu_i @ a_left) + (Wu_j @ a_right) )
        避免构造 (B, n, n, 2F') 的大张量.

        参数:
            u:   (B, n, F)   当前节点特征
            adj: (B, n, n)   邻接矩阵 (1=有边, 0=无边)

        返回:
            alpha: (B, n, n)  注意力权重, alpha[b,i,j] = alpha_ij
        """
        # W u: (B, n, F')
        Wu = self.W_attn(u)

        # (Wu @ a_left): (B, n) -- 节点 i 的贡献
        score_i = torch.einsum("bnf,f->bn", Wu, self.a_left)   # (B, n)
        # (Wu @ a_right): (B, n) -- 节点 j 的贡献
        score_j = torch.einsum("bnf,f->bn", Wu, self.a_right)  # (B, n)

        # beta_ij = score_i[i] + score_j[j], 广播为 (B, n, n)
        beta = score_i.unsqueeze(2) + score_j.unsqueeze(1)     # (B, n, n)
        beta = self.leaky_relu(beta)

        # masked softmax: 无边位置设为 -inf
        alpha = self._masked_softmax(beta, adj)
        return alpha

    # =================================================================
    # 消息计算 -- 公式 (28)
    # =================================================================
    def compute_messages(
        self,
        u: torch.Tensor,
        edge_feat: torch.Tensor,
        sigma2: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        论文公式 (28):
            m_ji = Psi(u_i, u_j, e_ij, sigma2; beta),  j in N(i)

        Psi 是 3 层 MLP, 输入 [u_i, u_j, e_ij, sigma2], 维度 2F+2.
        注意: u_i 是目标节点, u_j 是源节点 (消息从 j 发往 i).

        参数:
            u:         (B, n, F)    当前节点特征
            edge_feat: (B, n, n, 1) 标量边特征 e_ij = -h_i^T h_j
            sigma2:    (B,)         噪声方差
            adj:       (B, n, n)    邻接矩阵

        返回:
            msg: (B, n, n, F)  msg[b,i,j,:] = m_ji (从 j 到 i 的消息)
        """
        B, n, _ = u.shape

        # 构造每条边的输入: [u_i, u_j, e_ij, sigma2]
        u_i = u.unsqueeze(2).expand(B, n, n, self.F)   # (B, n, n, F) 目标节点
        u_j = u.unsqueeze(1).expand(B, n, n, self.F)   # (B, n, n, F) 源节点

        # e_ij: (B, n, n, 1) 已有
        # sigma2: (B,) -> (B, 1, 1, 1) -> (B, n, n, 1)
        sigma2_exp = sigma2.view(B, 1, 1, 1).expand(B, n, n, 1)

        # 拼接: (B, n, n, 2F+2)
        msg_input = torch.cat([u_i, u_j, edge_feat, sigma2_exp], dim=-1)

        # MLP: (B, n, n, 2F+2) -> (B, n, n, F)
        msg = self.msg_mlp(msg_input)

        # 对无效边置零
        msg = msg * adj.unsqueeze(-1)

        return msg

    # =================================================================
    # 消息聚合 + GRU 更新 -- 公式 (29)
    # =================================================================
    def aggregate_and_update(
        self,
        u: torch.Tensor,
        alpha: torch.Tensor,
        msg: torch.Tensor,
    ) -> torch.Tensor:
        """
        论文公式 (29):
            u_t_i = GRU(u_{t-1}_i, sum_{j in N(i)} alpha_ij * m_ji)

        参数:
            u:     (B, n, F)    上一层节点特征 (GRU hidden state)
            alpha: (B, n, n)    注意力权重
            msg:   (B, n, n, F) 消息张量

        返回:
            u_new: (B, n, F)    更新后的节点特征
        """
        B, n, _ = u.shape

        # 加权聚合: agg_i = sum_j alpha_ij * m_ji
        # alpha: (B,n,n) -> (B,n,n,1) * msg: (B,n,n,F) -> sum over j (dim=2)
        agg = (alpha.unsqueeze(-1) * msg).sum(dim=2)  # (B, n, F)

        # GRU 更新 (需要 reshape 为 2D)
        u_flat = u.reshape(B * n, self.F)
        agg_flat = agg.reshape(B * n, self.F)
        u_new_flat = self.gru(agg_flat, u_flat)  # GRU(input, hidden)
        u_new = u_new_flat.reshape(B, n, self.F)

        return u_new

    # =================================================================
    # 读出模块 -- 公式 (30)-(32)
    # =================================================================
    def readout(self, u: torch.Tensor) -> torch.Tensor:
        """
        论文公式 (30)-(31):
            r_i = MLP2(u_T_i)
            p_hat(x_i = s_k) = softmax_k(r_ik)

        参数:
            u: (B, n, F)  最终节点特征 (第 T 层输出)

        返回:
            logits: (B, n, S)  未经 softmax 的原始分数 (用于交叉熵损失)
        """
        return self.readout_mlp(u)  # (B, n, S)

    @staticmethod
    def hard_decision(logits: torch.Tensor) -> torch.Tensor:
        """
        论文公式 (32) 简化版: 逐节点 argmax.

        完整的公式 (32) 需要联合实部虚部:
            x_hat_i = argmax_{s_k1} { p_hat(x_i=s_k1) + p_hat(x_{i+MN}=s_k2) }
        这里先实现逐节点独立判决, 联合判决在 detector 层实现.

        参数:
            logits: (B, n, S)

        返回:
            决策索引: (B, n)  每个节点选择的星座点索引
        """
        return logits.argmax(dim=-1)

    def soft_estimate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        可选: 基于概率的软估计 (类似 AMP-GAT 的输出方式).
            x_hat_i = sum_k p_k * Q[k]

        参数:
            logits: (B, n, S)

        返回:
            x_hat: (B, n)  连续值软估计
        """
        p = torch.softmax(logits, dim=-1)                   # (B, n, S)
        Q = self.Q_R.to(logits.device).view(1, 1, -1)       # (1, 1, S)
        x_hat = (p * Q).sum(dim=-1)                          # (B, n)
        return x_hat

    # =================================================================
    # masked softmax 工具
    # =================================================================
    @staticmethod
    def _masked_softmax(score: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        对注意力分数做邻接掩码后 softmax.

        无边位置 (adj=0) 设为 -inf, softmax 后自然为 0.
        若某节点无任何邻居, softmax(-inf,...,-inf) 会产生 nan,
        此时回退为均匀分布 (实际中稀疏图应保证自环存在).

        参数:
            score: (B, n, n)  原始注意力分数
            adj:   (B, n, n)  邻接矩阵

        返回:
            alpha: (B, n, n)  归一化后的注意力权重
        """
        NEG_INF = -1e9
        mask = adj > 0
        s = torch.where(mask, score, torch.full_like(score, NEG_INF))
        alpha = torch.softmax(s, dim=-1)  # softmax over j (dim=-1)
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
        return alpha

    # =================================================================
    # 完整前向: T 层迭代
    # =================================================================
    def forward(
        self,
        u_0: torch.Tensor,
        adj: torch.Tensor,
        edge_feat: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        GAT-OTFS 核心前向传播: T 层迭代更新 + 读出.

        参数:
            u_0:       (B, n, F)    初始节点特征 (来自初始化模块)
            adj:       (B, n, n)    邻接矩阵
            edge_feat: (B, n, n, 1) 边特征
            sigma2:    (B,)         噪声方差

        返回:
            logits: (B, n, S)  读出层的原始分数 (未经 softmax)
                    可用于:
                    - CrossEntropyLoss 训练
                    - softmax 后做概率判决
                    - argmax 做硬判决
        """
        u = u_0  # (B, n, F)

        for t in range(self.T):
            # 1) 注意力系数 -- 公式 (27)
            alpha = self.compute_attention(u, adj)       # (B, n, n)

            # 2) 消息计算 -- 公式 (28)
            msg = self.compute_messages(u, edge_feat, sigma2, adj)  # (B, n, n, F)

            # 3) 聚合 + GRU 更新 -- 公式 (29)
            u = self.aggregate_and_update(u, alpha, msg)  # (B, n, F)

        # 4) 读出 -- 公式 (30)-(31)
        logits = self.readout(u)  # (B, n, S)

        return logits


# ===================================================================
# 单元测试
# ===================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # ---- 参数 ----
    B = 2       # batch size
    M, N = 4, 4
    n = 2 * M * N   # 32 (实值展开后)
    F_dim = 8
    T_layers = 3    # 测试时用少量层数加速

    print("=" * 60)
    print("GAT-OTFS Module 单元测试")
    print("=" * 60)

    # ---- 构造模拟输入 ----
    u_0 = torch.randn(B, n, F_dim)
    adj = (torch.rand(B, n, n) > 0.7).float()
    # 强制自环
    eye = torch.eye(n).unsqueeze(0).expand(B, -1, -1)
    adj = torch.maximum(adj, eye)
    edge_feat = torch.randn(B, n, n, 1)
    sigma2 = torch.tensor([0.1, 0.2])

    # ---- 创建模块 ----
    model = GATOTFSModule(F=F_dim, F_prime=16, T=T_layers, Nh1=64, Nh2=32)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 前向传播 ----
    logits = model(u_0, adj, edge_feat, sigma2)
    print(f"\nlogits shape: {logits.shape}")  # (2, 32, 2)
    assert logits.shape == (B, n, N_Q), f"logits shape 错误: {logits.shape}"

    # ---- 测试子模块 ----

    # (a) 注意力系数
    alpha = model.compute_attention(u_0, adj)
    print(f"alpha shape:  {alpha.shape}")     # (2, 32, 32)
    assert alpha.shape == (B, n, n)

    # 验证: 每行和 = 1 (在有邻居的行)
    row_sums = alpha.sum(dim=-1)              # (B, n)
    has_neighbor = (adj.sum(dim=-1) > 0)
    valid_sums = row_sums[has_neighbor]
    assert torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1e-5), \
        f"注意力行和不为1: {valid_sums[:5]}"
    print("  -> 注意力行和验证通过")

    # 验证: 无边位置 alpha=0
    no_edge = (adj == 0)
    assert (alpha[no_edge] == 0).all(), "无边位置 alpha 应为0"
    print("  -> 无边位置置零验证通过")

    # (b) 消息
    msg = model.compute_messages(u_0, edge_feat, sigma2, adj)
    print(f"msg shape:    {msg.shape}")       # (2, 32, 32, 8)
    assert msg.shape == (B, n, n, F_dim)

    # 验证: 无边位置消息为 0
    no_edge_4d = no_edge.unsqueeze(-1).expand_as(msg)
    assert (msg[no_edge_4d] == 0).all(), "无边位置消息应为0"
    print("  -> 消息掩码验证通过")

    # (c) 聚合 + GRU
    u_new = model.aggregate_and_update(u_0, alpha, msg)
    print(f"u_new shape:  {u_new.shape}")     # (2, 32, 8)
    assert u_new.shape == (B, n, F_dim)

    # (d) 读出
    logits_r = model.readout(u_new)
    print(f"logits shape: {logits_r.shape}")  # (2, 32, 2)
    assert logits_r.shape == (B, n, N_Q)

    # (e) 硬判决
    decisions = model.hard_decision(logits_r)
    print(f"decisions shape: {decisions.shape}, 取值范围: [{decisions.min()}, {decisions.max()}]")
    assert decisions.shape == (B, n)
    assert decisions.min() >= 0 and decisions.max() < N_Q

    # (f) 软估计
    x_hat = model.soft_estimate(logits_r)
    print(f"x_hat shape: {x_hat.shape}")
    assert x_hat.shape == (B, n)

    # ---- 梯度测试 ----
    print("\n--- 梯度测试 ---")
    model.zero_grad()
    logits_full = model(u_0, adj, edge_feat, sigma2)
    loss = logits_full.sum()
    loss.backward()

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    print(f"有梯度的参数数: {len(grad_norms)} / {sum(1 for _ in model.parameters())}")
    assert len(grad_norms) > 0, "没有参数收到梯度!"

    has_zero_grad = [k for k, v in grad_norms.items() if v == 0.0]
    if has_zero_grad:
        print(f"  警告: 以下参数梯度为零: {has_zero_grad}")
    else:
        print("  -> 所有参数梯度非零")

    # ---- 数值稳定性测试 ----
    print("\n--- 数值稳定性测试 ---")
    assert torch.isfinite(logits_full).all(), "logits 中存在 inf/nan"
    print("  -> logits 无 inf/nan")

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} 梯度中存在 inf/nan"
    print("  -> 所有梯度无 inf/nan")

    # ---- 交叉熵损失测试 ----
    print("\n--- 交叉熵损失测试 ---")
    # 模拟真实标签 (0 或 1)
    labels = torch.randint(0, N_Q, (B, n))
    ce_loss = nn.CrossEntropyLoss()(logits_full.reshape(-1, N_Q), labels.reshape(-1))
    print(f"  交叉熵损失: {ce_loss.item():.4f}")
    ce_loss.backward()
    print("  -> 交叉熵反向传播成功")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)