# 消融与测试计划（AMP-GAT 草图）

## 1) 消融矩阵

| ID | 变体 | 目的 | 预期结果 |
|---|---|---|---|
| A1 | AMP-GNN 基线 | 参考对照 | 形成基线 BER/loss/时延 |
| A2 | AMP-GAT（无 edge_attr） | 隔离注意力增益 | 若邻居加权有效，BER 优于或不劣于 A1 |
| A3 | AMP-GAT（+edge_attr） | 评估信道感知注意力 | 在更复杂信道下，BER/收敛优于 A2 |
| A4 | AMP-GAT（+damp） | 提升稳定性 | 发散尖峰更少，训练更平滑 |
| A5 | AMP-GAT（+多头） | 容量与代价权衡 | 计算开销适度上升下可能获得 BER 增益 |

## 2) 评估场景

### 信道/SNR 维度
- Single SNR training vs multi-SNR validation.
- Integer Doppler case (available now).
- Fractional Doppler case (if data available later).
- Sparse path vs denser path conditions.

### 数据集/实验协议
- Same AFDM dataset format for all variants (`x, y, H, sigma2`).
- Same train/val split and same random seed policy.
- Same training budget (`epochs`, `batch_size`, `lr`) unless the ablation explicitly studies those factors.

## 3) 验收标准

### 精度标准
- Primary: BER improvement over AMP-GNN baseline, or statistically equivalent BER with faster convergence.
- Secondary: lower or equal validation L2 loss.

### 稳定性标准
- No NaN/Inf during training.
- Variance states (`nu_x`, `nu_r`, `nu_z`) remain positive after clamping.
- Reproducible convergence trend in at least 2 independent runs.

### 复杂度标准
- Report per-epoch time and inference throughput.
- Complexity increase from attention must be measured and justified by BER/convergence gain.

## 4) 测试用例（设计层）

1. **形状一致性测试**
   - 验证中间张量满足 `(B,n)` / `(B,n,n)` / `(B,n,n,d_e)`。
2. **数值安全测试**
   - 每轮检查有限值与非负方差。
3. **退化注意力回归测试**
   - 当 `n_heads=1` 且注意力强制均匀时，输出趋势应接近非注意力聚合。
4. **训练冒烟测试**
   - 跑 1-2 个 epoch，loss 能从初始化下降。
5. **A/B 基线对比测试**
   - 在相同随机种子与预算下比较 AMP-GNN 与 AMP-GAT。

## 5) 报告模板

每次运行记录：
- Config ID
- `model_type`, `n_heads`, `damp`, `use_edge_attr`
- Best epoch
- Best `val_loss`
- Best `val_BER`
- Total train time
- 相对基线的 BER 提升

## 6) 推进建议
- Phase 1：先做 A1/A2（最小 AMP-GAT，不含边特征）。
- Phase 2：加入边特征与阻尼（A3/A4）。
- Phase 3：在计算预算约束下调优多头与 dropout（A5）。
