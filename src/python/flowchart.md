下面按你指定的“**单次迭代流程**”来做完整推导，并把公式和代码变量一一对齐。

**1) 统一模型与符号（主线实现）**

代码基准：
[amp_linear.py](/e:/code/AFDM_AMPNet_GAT/src/python/amp_linear.py)  
[amp_gnn_detector.py](/e:/code/AFDM_AMPNet_GAT/src/python/amp_gnn_detector.py)  
[gnn_module.py](/e:/code/AFDM_AMPNet_GAT/src/python/gnn_module.py)  
[amp_gat_detector.py](/e:/code/AFDM_AMPNet_GAT/src/python/amp_gat_detector.py)  
[gat_module.py](/e:/code/AFDM_AMPNet_GAT/src/python/gat_module.py)  
[graph_features.py](/e:/code/AFDM_AMPNet_GAT/src/python/graph_features.py)

实值等效系统（每个 batch 样本）：
$$
\mathbf y=\mathbf H\mathbf x+\mathbf w,\quad \mathbf y,\mathbf x\in\mathbb R^n,\ \mathbf H\in\mathbb R^{n\times n},\ n=2MN
$$
QPSK 在实值域上每维是二点星座：
$$
\mathcal Q_R=\{-1/\sqrt2,\ +1/\sqrt2\}
$$

主要变量（和代码同名）：
- \(\mathbf x^{(t)}_{\text{hat}}\) ↔ `x_hat`，形状 `(B,n)`
- \(\boldsymbol\nu_x^{(t)}\) ↔ `nu_x`，形状 `(B,n)`
- \(\mathbf z^{(t)}\), \(\boldsymbol\nu_z^{(t)}\) ↔ `z`, `nu_z`
- \(\mathbf r^{(t)}\), \(\boldsymbol\nu_r^{(t)}\) ↔ `r`, `nu_r`
- 图结构 `adj` 来自 \(G=H^\top H\) 的非零掩码

---

**2) 第 t 轮共同骨架：AMP 步（AMP-GNN/GAT 完全共享）**

来自 [amp_linear.py](/e:/code/AFDM_AMPNet_GAT/src/python/amp_linear.py) 的 `amp_linear_step`：

$$
\nu_{z,j}^{(t)}=\sum_i H_{j i}^2\,\nu_{x,i}^{(t)}+\varepsilon
$$
白话：把当前每个符号不确定度 \(\nu_x\) 通过信道功率 \(H^2\) 传播到观测域。  
意义：**方差传播**。
$$
z_j^{(t)}=(H x_{\text{hat}}^{(t)})_j-\frac{\nu_{z,j}^{(t)}}{\nu_{z,j}^{(t-1)}+\sigma^2}\left(y_j-z_j^{(t-1)}\right)
$$
白话：线性预测 \(Hx\) 再减一个与残差相关的修正项。  
意义：**Onsager/外差式残差校正**，抑制回声相关误差。
$$
\nu_{r,i}^{(t)}=
\left(\sum_j \frac{H_{j i}^2}{\nu_{z,j}^{(t)}+\sigma^2}\right)^{-1}
$$
白话：把观测端噪声+干扰不确定度映射回符号端，得到每个符号的等效方差。  
意义：**构造等效高斯先验宽度**。
$$
r_i^{(t)}=x_{\text{hat},i}^{(t)}+\nu_{r,i}^{(t)}
\sum_j H_{j i}\frac{y_j-z_j^{(t)}}{\nu_{z,j}^{(t)}+\sigma^2}
$$
白话：在旧估计上加一个“白化残差反投影”校正。  
意义：得到图网络去噪器输入 \((r,\nu_r)\)。

---

**3) 第 t 轮图步分支 A：AMP-GNN**

核心在 [gnn_module.py](E:/code/AFDM_AMPNet_GAT/src/python/gnn_module.py)。

先构造节点初始特征：
$$
\text{yh}_i = y^\top h_i,\quad \text{hh}_i=h_i^\top h_i,\quad
u_i^{(0)}=W_1[\text{yh}_i,\text{hh}_i]+b_1
$$
其中 \(h_i\) 是 \(H\) 的第 \(i\) 列（代码通过 `bmm` 得到）。

边权（图耦合强度）：
$$
e_{ij}=h_i^\top h_j=(H^\top H)_{ij}
$$

每层传播（共 `n_conv` 层）：
$$
\tilde m_i=\sum_{j\in\mathcal N(i)} [u_i,\ u_j,\ e_{ij}]
$$
$$
m_i=\text{MLP}(\tilde m_i)
$$
注意这里实现是“**先按邻居求和，再过 MLP**”，不是标准“每条边先 MLP 再求和”。

与 AMP 先验拼接：
$$
a_i=[r_i,\nu_{r,i}]
$$

GRU 更新：
$$
s_i\leftarrow \text{GRU}\big([m_i,a_i],\,s_i\big),\quad
u_i\leftarrow W_2 s_i+b_2
$$

读出概率并回到连续域：
$$
p_i=\text{softmax}(\text{Readout}(u_i))\in\mathbb R^{2}
$$
$$
\hat x_i^{(t+1)}=\sum_{q\in\mathcal Q_R} q\,p_i(q),\quad
\nu_{x,i}^{(t+1)}=\sum_{q\in\mathcal Q_R}(q-\hat x_i^{(t+1)})^2p_i(q)
$$

数学意义：
- MLP 聚合：学习近似后验中的“邻居干扰函数”
- GRU：把多层传播当成“有记忆的迭代推断器”
- softmax 读出：显式近似离散后验，再取一阶/二阶矩作为 AMP 下轮输入

---

**4) 第 t 轮图步分支 B：AMP-GAT**

核心在 [gat_module.py](/e:/code/AFDM_AMPNet_GAT/src/python/gat_module.py)。

初始特征直接含 AMP 先验：
$$
u_i^{(0)}=\text{InitProj}([y^\top h_i,\ h_i^\top h_i,\ r_i,\ \nu_{r,i}])
$$

每个 head \(h\)：
$$
q_i^h=W_q^h u_i,\quad k_j^h=W_k^h u_j,\quad v_j^h=W_v^h u_j
$$

注意力打分（代码是拆项高效实现）可写为：
$$
s_{ij}^h=\text{LeakyReLU}\left(
\langle a_q^h,q_i^h\rangle+\langle a_k^h,k_j^h\rangle
+\mathbf 1_{\text{edge}}\langle a_e^h, W_e^h e_{ij}\rangle
\right)
$$
其中 `edge_attr` 默认来自 [graph_features.py](/e:/code/AFDM_AMPNet_GAT/src/python/graph_features.py) 的 `gram_triplet`：
$$
e_{ij}=[h_i^\top h_j,\ \|h_i\|^2,\ \|h_j\|^2]
$$

masked softmax（只在邻接边上归一）：
$$
\alpha_{ij}^h=\frac{\exp(s_{ij}^h)\mathbf 1_{(i,j)\in E}}
{\sum_{k}\exp(s_{ik}^h)\mathbf 1_{(i,k)\in E}}
$$

head 消息：
$$
m_i^h=\sum_j \alpha_{ij}^h v_j^h,\quad
m_i=\frac1{H_{\text{heads}}}\sum_h m_i^h
$$

后续更新与读出和 GNN 同型：
$$
s_i\leftarrow \text{GRU}([m_i,a_i],s_i),\quad u_i\leftarrow \text{OutProj}(s_i)
$$
$$
\hat x_i^{(t+1)},\nu_{x,i}^{(t+1)} \text{ 仍由 softmax 后验矩得到}
$$

数学意义：
- \(\alpha_{ij}\) 是数据相关的邻居权重，能对不同样本自适应抑制/放大干扰边
- 多头等价于并行子空间下的多组“干扰解释器”

---

**5) 外层同轮并排对比 + 阻尼**

两者检测器外层都是：
$$
(r^{(t)},\nu_r^{(t)})=\text{AMPLinear}(x_{\text{hat}}^{(t)},\nu_x^{(t)})
$$
$$
(x_{\text{hat}}^{(t+1)},\nu_x^{(t+1)})=\text{GraphDenoiser}(r^{(t)},\nu_r^{(t)})
$$

AMP-GAT 在 [amp_gat_detector.py](/e:/code/AFDM_AMPNet_GAT/src/python/amp_gat_detector.py) 多了阻尼：
$$
x_{\text{hat}}^{(t+1)}\leftarrow \lambda x_{\text{new}}+(1-\lambda)x_{\text{hat}}^{(t)}
$$
$$
\nu_x^{(t+1)}\leftarrow \lambda \nu_{\text{new}}+(1-\lambda)\nu_x^{(t)}
$$
\(\lambda=\text{damp}\)。  
意义：减小迭代振荡，提升收敛稳定性（代价是可能变慢）。

至少 5 个实现差异及数学后果：
1. GNN 初始特征不含 \((r,\nu_r)\)，GAT 初始特征包含。后果：GAT 更早注入 AMP 先验。  
2. GNN 是固定求和聚合，GAT 是 \(\alpha_{ij}\) 自适应加权。后果：GAT 表达力更强。  
3. GAT 可显式用 `edge_attr` 三元组，GNN 仅用标量 Gram（并扩展拼接）。后果：GAT 可编码更丰富边几何。  
4. GAT 有多头，GNN 单通道。后果：GAT 可并行建模多种干扰模式。  
5. AMP-GAT 额外阻尼，AMP-GNN 无。后果：GAT 默认更重视稳定迭代控制。  
6. GAT 打分实现避免构造巨大拼接张量。后果：同等 \(n\) 下显存更友好。

---

**6) 单轮迭代伪代码（回看源码用）**

> Input at round t: x_hat^t, nu_x^t
>
> Shared AMP step:
>
> nu_z^t = (H.^2) * nu_x^t 
>
> z^t    = H*x_hat^t - (nu_z^t/(nu_z^(t-1)+sigma2)) .* (y - z^(t-1)) 
>
> nu_r^t = 1 ./ ((H.^2)^T * (1./(nu_z^t+sigma2))) 
>
> r^t    = x_hat^t + nu_r^t .* (H^T * ((y-z^t)./(nu_z^t+sigma2)))

```text
Graph step:
  if AMP-GNN:
    u^0 <- from [y^T h_i, h_i^T h_i]
    repeat n_conv:
      m_i <- MLP(sum_j adj_ij [u_i,u_j,e_ij])
      s_i <- GRU([m_i, r_i, nu_r_i], s_i)
      u_i <- linear(s_i)
  if AMP-GAT:
    u^0 <- from [y^T h_i, h_i^T h_i, r_i, nu_r_i]
    repeat n_conv:
      for each head h:
        score_ij^h <- f(q_i^h, k_j^h, edge_attr_ij)
        alpha_ij^h <- masked_softmax(score_ij^h, adj)
        m_i^h <- sum_j alpha_ij^h v_j^h
      m_i <- average_h m_i^h
      s_i <- GRU([m_i, r_i, nu_r_i], s_i)
      u_i <- linear(s_i)

Readout (both):
  p_i <- softmax(readout(u_i))
  x_hat_i^(t+1) <- E_q[q | p_i]
  nu_x_i^(t+1)  <- Var_q[q | p_i]

Optional (AMP-GAT detector):
  damping mix with previous x_hat^t, nu_x^t
```

一句话总结本质：**AMP-GNN 和 AMP-GAT 都是在学习“AMP 的非线性去噪器” \(g_\theta(r,\nu_r)\)，区别只在于去噪器参数化是“MLP聚合”还是“注意力聚合”。**