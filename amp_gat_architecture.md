# AMP+GAT 架构（AFDM）

## 1. 端到端数据流

### 输入
- `y`：实值接收向量，形状 `(B, n)`，其中 `n = 2MN`（实值展开后）。
- `H`：实值等效信道矩阵，形状 `(B, n, n)`。
- `sigma2`：噪声方差，形状 `(B,)`。
- `adj`：邻接掩码，形状 `(B, n, n)`。
- `edge_attr`（可选）：边特征张量，形状 `(B, n, n, d_e)`。

### 迭代循环（`t = 1..T`）
状态变量：
- `x_hat^{t-1}`, `nu_x^{t-1}`：上一轮后验均值/方差估计。
- `z^{t-1}`, `nu_z^{t-1}`：AMP 隐变量状态。

每轮流程：
1. **AMP 步**
   - 输入：`(x_hat^{t-1}, nu_x^{t-1}, z^{t-1}, nu_z^{t-1}, y, H, sigma2)`
   - 输出：`(r^t, nu_r^t, z^t, nu_z^t)`
2. **GAT 步**
   - 输入：`(r^t, nu_r^t, H, adj, edge_attr)`
   - 输出：`(x_new^t, nu_new^t)`
3. **可选阻尼**
   - `x_hat^t = damp * x_new^t + (1 - damp) * x_hat^{t-1}`
   - `nu_x^t = damp * nu_new^t + (1 - damp) * nu_x^{t-1}`
4. 进入下一轮。

### 输出
- `x_hat^T`：最终实值估计。

## 2. GAT 注意力定义（掩码稀疏注意力）

对每条有效边 `(i, j)`（满足 `adj(i,j)=1`）：

1. 节点投影：
- `u_i' = W_u u_i`
- `u_j' = W_u u_j`

2. 边投影（启用边特征时）：
- `e_ij' = W_e edge_ij`

3. 注意力打分：
- `score_ij = LeakyReLU(a^T [u_i' || u_j' || e_ij'])`

4. 邻域归一化（掩码 softmax）：
- `alpha_ij = softmax_j(score_ij), j in N(i)`

5. 消息聚合：
- `m_i = sum_{j in N(i)} alpha_ij * V u_j`

多头策略（默认）：
- `n_heads = 2`
- 头融合方式固定为 `mean`（避免维度膨胀，并简化 AMP 接口对接）。

## 3. 节点/边特征策略

### 节点初始化
- 必选 AMP 先验特征：
  - `a_i = [r_i, nu_r_i]`
- 保留当前基线信道感知特征：
  - `c_i = [y^T h_i, h_i^T h_i]`
- 节点初始输入：
  - `node_i_input = [c_i || a_i]`

### 边特征（默认）
- 推荐默认 `mode='gram_triplet'`：
  - `edge_ij = [h_i^T h_j, ||h_i||^2, ||h_j||^2]`
- 最简回退 `mode='gram'`：
  - `edge_ij = [h_i^T h_j]`

## 4. 与两篇论文的映射关系

继承 AMP-GNN 论文部分：
- AMP 产生逐符号高斯先验 `(r, nu_r)`。
- 线性估计与图模块交替迭代。

引入 GAT-OTFS 论文部分：
- 邻居消息采用自适应注意力加权。
- 在稀疏图上进行掩码聚合。

融合结果：
- AMP 保留模型驱动先验细化能力。
- GAT 用关系感知加权替代原本统一/求和式聚合。

## 5. 稳定性默认策略
- 默认 `damp = 0.7`。
- 所有方差类变量裁剪到下界：`nu_x, nu_r, nu_z >= 1e-10`。
- 若某节点无有效邻居，掩码 softmax 需有有限值回退策略。
