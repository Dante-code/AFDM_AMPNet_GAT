# AMP+GAT 混合检测草图

## 目标
本草图用于在现有 AFDM AMP-GNN 代码基础上，给出一份“决策完整”的 AMP（外环）与 GAT（图推理内环）融合蓝图，不修改 `backup/python/*` 下的生产代码。

## 范围
- 仅包含设计与伪代码。
- 不改动 `backup/python/*` 的实现。
- 输出文件仅位于 `./draft`。

## 当前方案与目标方案对比

| 项目 | 当前 AMP-GNN | 目标 AMP-GAT |
|---|---|---|
| 外层迭代 | AMP <-> GNN | AMP <-> GAT |
| 图聚合方式 | MLP 求和聚合 | 掩码稀疏注意力聚合 |
| 邻居加权 | 隐式/共享权重 | 按边学习的注意力权重 |
| 图输入 | `adj` | `adj` + 可选 `edge_attr` |
| 阻尼 | Python 路径无显式阻尼 | 显式可配置 `damp` |
| 配置键 | `n_iter, n_u, ...` | 新增 `model_type, n_heads, attn_dropout, damp, use_edge_attr` |

## 与现有工程的关系
- 现有基线模块：
  - `backup/python/amp_linear.py`
  - `backup/python/gnn_module.py`
  - `backup/python/amp_gnn_detector.py`
  - `backup/python/train_afdm.py`
  - `backup/python/config/train_config.yaml`
- 本草图说明这些模块在 AMP-GAT 下应如何演进，同时保持 AMP 数学形式与 AFDM 数据格式（`x,y,H,sigma2`）兼容。

## 阅读顺序
1. `amp_gat_architecture.md`
2. `module_change_map.md`
3. `pseudocode_amp_gat_detector.md`
4. `pseudocode_gat_module.md`
5. `pseudocode_training_pipeline.md`
6. `config_schema_amp_gat.yaml`
7. `ablation_and_test_plan.md`
