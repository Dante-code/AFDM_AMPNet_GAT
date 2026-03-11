"""
GAT-OTFS 顶层检测器 -- 端到端封装

将 gat_otfs_init.py (初始化模块) 和 gat_otfs_module.py (核心网络) 组合为
一个完整的 nn.Module, 实现从原始输入 (y, H, sigma2) 到检测结果的端到端推理.

对应论文 Algorithm 1 的完整流程:
    Input:  y, H, sigma2, (信道参数用于构建邻接表)
    Step 1: 初始化 -- 构建 adj, edge_feat, n_i, u_0
    Step 2: T 层迭代更新 -- 注意力 + 消息MLP + GRU
    Step 3: 读出 -- MLP + softmax
    Output: x_hat (硬判决) 或 logits (用于训练)

与 GitHub 现有 AMP-GAT 检测器 (amp_gat_detector.py) 的区别:
    - 无 AMP 外层迭代循环
    - 无阻尼 (damp) 机制
    - 直接输入 (y, H, sigma2), 不需要 (r, nu_r)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from gat_otfs_init import (
    build_adjacency,
    build_edge_features,
    extract_node_status,
    NodeInitFFN,
)
from gat_otfs_module import GATOTFSModule, N_Q, Q_R


class GATOTFSDetector(nn.Module):
    """
    GAT-OTFS 检测器: 论文 Algorithm 1 的完整实现.

    参数:
        F:        节点特征维度 (论文默认 8)
        F_prime:  注意力升维维度 (论文默认 16)
        T:        迭代层数 (论文默认 10)
        Nh1:      MLP 第一隐层 (论文默认 64)
        Nh2:      MLP 第二隐层 (论文默认 32)
        S:        星座点实部集合大小 (QPSK=2)
        adj_eps:  邻接矩阵非零判定阈值
        add_self_loop: 邻接矩阵是否添加自环
    """

    def __init__(
        self,
        F: int = 8,
        F_prime: int = 16,
        T: int = 10,
        Nh1: int = 64,
        Nh2: int = 32,
        S: int = N_Q,
        adj_eps: float = 1e-8,
        add_self_loop: bool = True,
    ):
        super().__init__()
        self.F = F
        self.T = T
        self.S = S
        self.adj_eps = adj_eps
        self.add_self_loop = add_self_loop

        # 初始化模块: 公式 (26)
        self.node_init = NodeInitFFN(F=F)

        # 核心网络: 公式 (27)-(31)
        self.gat_core = GATOTFSModule(
            F=F,
            F_prime=F_prime,
            T=T,
            Nh1=Nh1,
            Nh2=Nh2,
            S=S,
        )

        # 星座点 (用于 BER 计算等)
        self.register_buffer("Q_R", torch.tensor(Q_R, dtype=torch.float32))

    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        端到端前向传播.

        参数:
            y:      (B, n)     接收信号, n = 2MN
            H:      (B, n, n)  实值等效信道矩阵
            sigma2: (B,)       噪声方差

        返回:
            logits: (B, n, S)  读出层原始分数
                    训练时传入 CrossEntropyLoss
                    推理时用 argmax 做硬判决
        """
        # --- Step 1: 初始化 ---
        # 邻接矩阵: 公式 (16)/(25)
        adj = build_adjacency(H, eps=self.adj_eps, add_self_loop=self.add_self_loop)

        # 边特征: 公式 (15)
        edge_feat = build_edge_features(H)

        # 节点状态 + 初始投影: 公式 (14) + (26)
        n_i = extract_node_status(y, H, sigma2)
        u_0 = self.node_init(n_i)

        # --- Step 2 + 3: T 层迭代 + 读出 ---
        logits = self.gat_core(u_0, adj, edge_feat, sigma2)

        return logits

    # =================================================================
    # 推理辅助方法
    # =================================================================
    def detect(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        推理接口: 返回硬判决结果 (星座点索引).

        返回:
            decisions: (B, n)  每个节点的判决索引, 0 或 1 (QPSK)
        """
        logits = self.forward(y, H, sigma2)
        return self.gat_core.hard_decision(logits)

    def detect_symbols(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        推理接口: 返回判决后的星座点实值.

        返回:
            x_hat: (B, n)  判决后的实值符号
        """
        decisions = self.detect(y, H, sigma2)          # (B, n), 索引
        Q = self.Q_R.to(y.device)                      # (S,)
        x_hat = Q[decisions]                            # (B, n)
        return x_hat

    def detect_joint(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        sigma2: torch.Tensor,
        MN: int,
    ) -> torch.Tensor:
        """
        论文公式 (32) 的联合判决: 同时考虑实部和虚部.

        x_hat_i = argmax_{s_k1, s_k2}
                  { p_hat(x_i = s_k1) + p_hat(x_{i+MN} = s_k2) }

        参数:
            y, H, sigma2: 同 forward
            MN: 复数符号维度 (n = 2*MN)

        返回:
            x_hat: (B, n)  联合判决后的实值符号
        """
        logits = self.forward(y, H, sigma2)    # (B, 2MN, S)
        probs = torch.softmax(logits, dim=-1)  # (B, 2MN, S)

        B = y.shape[0]
        Q = self.Q_R.to(y.device)
        x_hat = torch.zeros(B, 2 * MN, device=y.device)

        # 对每个复数符号 i, 联合判决实部 (节点 i) 和虚部 (节点 i+MN)
        prob_re = probs[:, :MN, :]      # (B, MN, S) 实部概率
        prob_im = probs[:, MN:, :]      # (B, MN, S) 虚部概率

        # 联合概率: p_re[s1] + p_im[s2] 对所有 (s1, s2) 组合
        # prob_re: (B, MN, S, 1), prob_im: (B, MN, 1, S)
        joint = prob_re.unsqueeze(-1) + prob_im.unsqueeze(-2)  # (B, MN, S, S)

        # argmax 得到 (s1_idx, s2_idx)
        flat_idx = joint.reshape(B, MN, -1).argmax(dim=-1)    # (B, MN)
        s1_idx = flat_idx // self.S
        s2_idx = flat_idx % self.S

        x_hat[:, :MN] = Q[s1_idx]
        x_hat[:, MN:] = Q[s2_idx]

        return x_hat

    # =================================================================
    # BER 计算
    # =================================================================
    @staticmethod
    def compute_ber(
        x_true: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> float:
        """
        计算比特错误率 (BER).

        QPSK 实值编码下, 每个实数节点对应 1 bit:
            x > 0 -> bit 1,  x <= 0 -> bit 0

        参数:
            x_true: (B, n)  真实实值符号
            x_hat:  (B, n)  检测结果实值符号

        返回:
            ber: float  比特错误率
        """
        bits_true = (x_true > 0).long()
        bits_hat = (x_hat > 0).long()
        errors = (bits_true != bits_hat).sum().item()
        total = x_true.numel()
        return errors / total if total > 0 else 0.0

    # =================================================================
    # 标签生成工具
    # =================================================================
    @staticmethod
    def symbols_to_labels(x_true: torch.Tensor) -> torch.Tensor:
        """
        将真实符号转换为分类标签 (用于 CrossEntropyLoss).

        QPSK 实部: Q_R = [-1/sqrt(2), +1/sqrt(2)]
            x <= 0 -> label 0 (对应 -1/sqrt(2))
            x >  0 -> label 1 (对应 +1/sqrt(2))

        参数:
            x_true: (B, n)  真实实值符号

        返回:
            labels: (B, n)  long 类型, 取值 {0, 1}
        """
        return (x_true > 0).long()


# ===================================================================
# 单元测试
# ===================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    B = 2
    M, N = 4, 4
    n = 2 * M * N  # 32
    MN = M * N     # 16

    print("=" * 60)
    print("GAT-OTFS Detector 端到端测试")
    print("=" * 60)

    # ---- 模拟输入 ----
    y = torch.randn(B, n)
    H = torch.randn(B, n, n) * 0.1
    sigma2 = torch.tensor([0.1, 0.2])

    # 模拟真实符号 (QPSK)
    q = 1.0 / np.sqrt(2)
    x_true = torch.tensor(np.random.choice([-q, q], size=(B, n))).float()

    # ---- 创建检测器 (用小参数加速测试) ----
    detector = GATOTFSDetector(F=8, F_prime=16, T=3, Nh1=32, Nh2=16)
    n_params = sum(p.numel() for p in detector.parameters())
    print(f"\n模型参数量: {n_params:,}")

    # ---- 测试 forward ----
    print("\n--- forward 测试 ---")
    logits = detector(y, H, sigma2)
    print(f"logits shape: {logits.shape}")  # (2, 32, 2)
    assert logits.shape == (B, n, N_Q)
    assert torch.isfinite(logits).all()
    print("  -> 通过")

    # ---- 测试 detect (硬判决) ----
    print("\n--- detect 测试 ---")
    decisions = detector.detect(y, H, sigma2)
    print(f"decisions shape: {decisions.shape}, 范围: [{decisions.min()}, {decisions.max()}]")
    assert decisions.shape == (B, n)
    print("  -> 通过")

    # ---- 测试 detect_symbols ----
    print("\n--- detect_symbols 测试 ---")
    x_hat = detector.detect_symbols(y, H, sigma2)
    print(f"x_hat shape: {x_hat.shape}")
    unique_vals = x_hat.unique().sort()[0]
    print(f"x_hat 取值: {unique_vals.tolist()}")
    assert set(x_hat.unique().tolist()).issubset({-q, q})
    print("  -> 通过 (输出为合法 QPSK 星座点)")

    # ---- 测试 detect_joint (联合判决) ----
    print("\n--- detect_joint 测试 ---")
    x_hat_joint = detector.detect_joint(y, H, sigma2, MN=MN)
    print(f"x_hat_joint shape: {x_hat_joint.shape}")
    assert x_hat_joint.shape == (B, n)
    print("  -> 通过")

    # ---- 测试 BER ----
    print("\n--- BER 测试 ---")
    ber = detector.compute_ber(x_true, x_hat)
    print(f"BER (随机模型 vs 真实): {ber:.4f}")
    ber_perfect = detector.compute_ber(x_true, x_true)
    assert ber_perfect == 0.0, "相同输入 BER 应为 0"
    print(f"BER (自身 vs 自身):     {ber_perfect:.4f}")
    print("  -> 通过")

    # ---- 测试标签生成 ----
    print("\n--- 标签生成测试 ---")
    labels = detector.symbols_to_labels(x_true)
    print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
    assert labels.shape == (B, n)
    assert labels.dtype == torch.long
    assert set(labels.unique().tolist()) == {0, 1}
    print("  -> 通过")

    # ---- 测试交叉熵训练流程 ----
    print("\n--- 交叉熵训练流程测试 ---")
    detector.train()
    optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)
    labels = detector.symbols_to_labels(x_true)

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        logits = detector(y, H, sigma2)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, N_Q), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  5 步损失: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0], "损失应在下降"
    print("  -> 损失下降验证通过")

    # ---- 参数统计 ----
    print("\n--- 各子模块参数分布 ---")
    init_params = sum(p.numel() for p in detector.node_init.parameters())
    core_params = sum(p.numel() for p in detector.gat_core.parameters())
    print(f"  初始化模块: {init_params:,}")
    print(f"  核心网络:   {core_params:,}")
    print(f"  总计:       {n_params:,}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)