# 模块改造映射（基线 -> AMP-GAT 草图）

本文件用于对照当前代码，标注 AMP-GAT 所需改造项。

## 标记说明
- **必须改**：实现 AMP-GAT 的必要改动。
- **可选改**：增强项，第一版可以不做。
- **暂不改**：第一阶段保持现状。

---

## 1) `backup/python/amp_gnn_detector.py`

### 必须改
- 新增检测器类设计：`AMPGATDetector`。
- 新 `forward` 接口：
  - `forward(y, H, sigma2, adj, edge_attr=None) -> x_hat`
- 保留 AMP 状态变量：
  - `x_hat, nu_x, z, nu_z`
- 将 `self.gnn(...)` 替换为 `self.gat(...)`。
- 在 **GAT 输出之后** 插入阻尼：
  - `x_hat <- damp*x_new + (1-damp)*x_prev`
  - `nu_x <- damp*nu_new + (1-damp)*nu_prev`

### 可选改
- 给 `z, nu_z` 增加 AMP 内部阻尼。
- 在调试模式返回诊断量（`x_hist`, `attn_stats`）。

### 暂不改
- `compute_ber`, `compute_l2_loss` 第一版可保持不变。

---

## 2) `backup/python/gnn_module.py`

### 必须改
- 逻辑上拆分职责：
  - 图特征构建（`build_adjacency`, `build_edge_attr`）
  - 消息核心（`GATModule`）
- 将 `msg_mlp(sum_j(...))` 聚合替换为注意力聚合。
- 支持 `n_heads`, `attn_dropout`, `use_edge_attr`。

### 可选改
- 增加边状态更新网络（每层边特征细化）。
- 每层 GAT 块增加残差/层归一化。

### 暂不改
- 读出语义（`softmax -> x_hat, nu_x`）先与当前 QPSK 实值路径保持一致。

---

## 3) `backup/python/amp_linear.py`

### 必须改
- 第一版 AMP-GAT 不改 AMP 方程本体。

### 可选改
- 返回更多调试中间量（如 `residual`, `denom`）。
- 增加 AMP 侧阻尼参数。

### 暂不改
- `amp_linear_step` 数学与 API 先保持兼容。

---

## 4) `backup/python/train_afdm.py`

### 必须改
- 基于 `model_type` 增加模型工厂：
  - `amp_gnn`（基线）
  - `amp_gat`（新）
- 当 `model_type=amp_gat` 时构建并传入 `edge_attr` batch。
- 保留 `adj` 构建路径，作为共享图工具。
- 在 CSV 记录新增模型超参数。

### 可选改
- 增加混合损失模式（`L2 + lambda*CE`）。
- 每轮记录注意力稀疏度诊断信息。

### 暂不改
- 数据加载路径（`x,y,H,sigma2`）保持不变。

---

## 5) `backup/python/config/train_config.yaml`

### 必须改
- 在 `model` 下新增字段：
  - `model_type`
  - `n_heads`
  - `attn_dropout`
  - `damp`
  - `use_edge_attr`
  - `edge_attr_mode`

### 可选改
- `loss_type`, `ce_weight`, `debug_attention`。

### 暂不改
- 现有训练字段（`batch_size`, `lr`, scheduler 等）可保持。

---

## 6) 新增工具拆分（推荐结构）

### 必须改（设计层）
- 实现阶段建议新增图工具模块：
  - `graph_features.py`
  - `build_adjacency(H, eps)`
  - `build_edge_attr(H, mode='gram')`

### 可选改
- 当信道矩阵重复出现时，对图特征做 mini-batch 缓存。

---

## 7) 依赖与兼容性说明
- 保持实值张量约定（`n = 2MN`），与当前 AFDM 预处理兼容。
- AMP 方程不改动，降低回归风险。
- 训练循环和 checkpoint 命名风格保持一致，便于与基线对比。
