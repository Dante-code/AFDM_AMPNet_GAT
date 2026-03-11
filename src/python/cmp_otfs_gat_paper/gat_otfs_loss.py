"""
GAT-OTFS 损失函数与评估工具 -- 对应论文公式 (33)

论文使用交叉熵损失:
    Loss = -sum_i sum_{x_i in Omega} p(x_i) * log p_hat(x_i)

其中 p(x_i) 是真实符号的 one-hot 分布, p_hat(x_i) 是模型输出的 softmax 概率.

与 GitHub 现有 AMP-GAT 代码的区别:
    - AMP-GAT 使用 L2 损失: loss = mean((x_true - x_hat)^2)
    - GAT-OTFS 使用交叉熵: 模型输出 logits, 真实符号映射为类别索引
    - 本文件同时提供 L2 辅助损失用于监控

QPSK 实值编码:
    星座点实部 Q_R = {-1/sqrt(2), +1/sqrt(2)}
    label 0 -> -1/sqrt(2)  (即 x <= 0)
    label 1 -> +1/sqrt(2)  (即 x >  0)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

Q_R = np.array([-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.float32)
N_Q = len(Q_R)


# ============================================================
# 1) 标签转换
# ============================================================
def symbols_to_labels(x_true: torch.Tensor) -> torch.Tensor:
    """
    将 QPSK 实值符号转换为分类标签.

    QPSK 实部: Q_R = [-1/sqrt(2), +1/sqrt(2)]
        x <= 0 -> label 0
        x >  0 -> label 1

    参数:
        x_true: (B, n) 真实实值符号

    返回:
        labels: (B, n) long 类型, 取值 {0, 1}
    """
    return (x_true > 0).long()


def labels_to_symbols(labels: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """
    将类别标签转换回实值符号.

    参数:
        labels: (B, n) long 类型, 取值 {0, 1}
        device: 目标设备

    返回:
        x: (B, n) 实值符号
    """
    Q = torch.tensor(Q_R, dtype=torch.float32)
    if device is not None:
        Q = Q.to(device)
    return Q[labels]


# ============================================================
# 2) 交叉熵损失 -- 公式 (33)
# ============================================================
class GATOTFSCELoss(nn.Module):
    """
    论文公式 (33) 的交叉熵损失.

    Loss = -sum_i sum_k p(x_i = s_k) * log p_hat(x_i = s_k)

    对于硬标签 (p 为 one-hot), 这等价于标准 CrossEntropyLoss.

    可选: 附加 L2 辅助损失用于监控收敛.
    """

    def __init__(self, l2_weight: float = 0.0, label_smoothing: float = 0.0):
        """
        参数:
            l2_weight:       L2 辅助损失权重 (0 表示纯交叉熵)
            label_smoothing: 标签平滑系数 (0 表示不平滑)
        """
        super().__init__()
        self.l2_weight = l2_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.register_buffer("Q_R", torch.tensor(Q_R, dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        x_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        参数:
            logits: (B, n, S)  模型输出的原始分数 (未经 softmax)
            x_true: (B, n)     真实实值符号

        返回:
            loss:    标量, 总损失
            metrics: 字典, 包含各项损失分量
        """
        B, n, S = logits.shape

        # 交叉熵
        labels = symbols_to_labels(x_true)                     # (B, n)
        ce_loss = self.ce(logits.reshape(-1, S), labels.reshape(-1))

        metrics = {"ce_loss": ce_loss.item()}

        # 可选 L2 辅助损失
        if self.l2_weight > 0:
            probs = torch.softmax(logits, dim=-1)              # (B, n, S)
            Q = self.Q_R.to(logits.device).view(1, 1, -1)     # (1, 1, S)
            x_hat = (probs * Q).sum(dim=-1)                    # (B, n)
            l2_loss = ((x_true - x_hat) ** 2).mean()
            total = ce_loss + self.l2_weight * l2_loss
            metrics["l2_loss"] = l2_loss.item()
            metrics["total_loss"] = total.item()
            return total, metrics

        metrics["total_loss"] = ce_loss.item()
        return ce_loss, metrics


# ============================================================
# 3) BER 计算
# ============================================================
def compute_ber_from_logits(
    logits: torch.Tensor,
    x_true: torch.Tensor,
) -> float:
    """
    从模型输出 logits 计算 BER.

    流程: logits -> argmax -> 星座点索引 -> 与真实标签比较

    参数:
        logits: (B, n, S)  模型输出
        x_true: (B, n)     真实实值符号

    返回:
        ber: float
    """
    pred_labels = logits.argmax(dim=-1)           # (B, n)
    true_labels = symbols_to_labels(x_true)       # (B, n)
    errors = (pred_labels != true_labels).sum().item()
    total = x_true.numel()
    return errors / total if total > 0 else 0.0


def compute_ber_from_symbols(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
) -> float:
    """
    从实值符号计算 BER (兼容 AMP-GAT 的输出).

    参数:
        x_hat:  (B, n) 检测结果
        x_true: (B, n) 真实符号

    返回:
        ber: float
    """
    bits_hat = (x_hat > 0).long()
    bits_true = (x_true > 0).long()
    errors = (bits_hat != bits_true).sum().item()
    total = x_true.numel()
    return errors / total if total > 0 else 0.0


# ============================================================
# 4) L2 损失 (纯辅助, 兼容现有代码)
# ============================================================
def compute_l2_loss(x_true: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """兼容现有 AMP-GNN 代码的 L2 损失."""
    return ((x_true - x_hat) ** 2).mean()


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    B, n, S = 4, 32, N_Q
    q = 1.0 / np.sqrt(2)

    print("=" * 50)
    print("GAT-OTFS Loss 单元测试")
    print("=" * 50)

    # 模拟数据
    x_true = torch.tensor(np.random.choice([-q, q], size=(B, n))).float()
    logits = torch.randn(B, n, S, requires_grad=True)

    # ---- 标签转换 ----
    labels = symbols_to_labels(x_true)
    print(f"labels shape: {labels.shape}, 取值: {labels.unique().tolist()}")
    x_back = labels_to_symbols(labels)
    assert torch.allclose(x_true, x_back, atol=1e-6), "标签往返转换失败"
    print("  -> 标签往返验证通过")

    # ---- 纯交叉熵损失 ----
    loss_fn = GATOTFSCELoss(l2_weight=0.0)
    loss, metrics = loss_fn(logits, x_true)
    print(f"\n纯 CE 损失: {loss.item():.4f}")
    print(f"  metrics: {metrics}")
    loss.backward()
    assert logits.grad is not None, "logits 应有梯度"
    print("  -> 反向传播通过")

    # ---- 混合损失 (CE + L2) ----
    logits2 = torch.randn(B, n, S, requires_grad=True)
    loss_fn2 = GATOTFSCELoss(l2_weight=0.5)
    loss2, metrics2 = loss_fn2(logits2, x_true)
    print(f"\n混合损失: {loss2.item():.4f}")
    print(f"  metrics: {metrics2}")
    assert "l2_loss" in metrics2
    loss2.backward()
    print("  -> 混合损失反向传播通过")

    # ---- BER 计算 ----
    # 完美 logits (真实标签对应位置给高分)
    perfect_logits = torch.zeros(B, n, S)
    for b in range(B):
        for i in range(n):
            perfect_logits[b, i, labels[b, i]] = 10.0
    ber_perfect = compute_ber_from_logits(perfect_logits, x_true)
    assert ber_perfect == 0.0, f"完美 logits BER 应为 0, 得到 {ber_perfect}"
    print(f"\n完美 logits BER: {ber_perfect}")

    # 随机 logits
    ber_random = compute_ber_from_logits(torch.randn(B, n, S), x_true)
    print(f"随机 logits BER: {ber_random:.4f}")
    print("  -> BER 测试通过")

    # ---- 带 label_smoothing ----
    loss_fn3 = GATOTFSCELoss(l2_weight=0.0, label_smoothing=0.1)
    loss3, _ = loss_fn3(torch.randn(B, n, S, requires_grad=True), x_true)
    print(f"\n标签平滑 CE 损失: {loss3.item():.4f}")

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
