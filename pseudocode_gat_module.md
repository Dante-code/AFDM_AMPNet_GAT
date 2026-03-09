# 伪代码：GAT 模块与边特征

## 1) 边特征构建器

### 接口
```python
def build_edge_attr(H: Tensor, mode: str = "gram_triplet", eps: float = 1e-8) -> Tensor:
    """
    H: (B, n, n)
    returns edge_attr: (B, n, n, d_e)
    """
```

### 伪代码
```python
def build_edge_attr(H, mode="gram_triplet", eps=1e-8):
    # Gram matrix: G_ij = h_i^T h_j
    G = bmm(H.transpose(1, 2), H)   # (B, n, n)

    if mode == "gram":
        return G.unsqueeze(-1)      # d_e=1

    if mode == "gram_triplet":
        # norms
        hi2 = diagonal(G, dim1=1, dim2=2)          # (B, n)
        hj2 = hi2
        hi2_expand = hi2.unsqueeze(2).expand_as(G) # (B, n, n)
        hj2_expand = hj2.unsqueeze(1).expand_as(G) # (B, n, n)
        edge = stack([G, hi2_expand, hj2_expand], dim=-1)  # d_e=3
        return edge

    raise ValueError("unsupported edge_attr mode")
```

## 2) GAT 模块接口

```python
class GATModule(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_u: int = 12,
        n_h: int = 12,
        n_conv: int = 2,
        n_heads: int = 2,
        attn_dropout: float = 0.0,
        use_edge_attr: bool = True,
        edge_attr_dim: int = 3,
        readout_hidden: int = 16
    ):
        ...

    def forward(
        self,
        y: Tensor, H: Tensor,
        r: Tensor, nu_r: Tensor,
        adj: Tensor,
        edge_attr: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:   # x_hat, nu_x with shape (B,n)
        ...
```

## 3) 节点初始化

融合信道特征与 AMP 先验：
- channel summary: `c_i = [y^T h_i, h_i^T h_i]`
- AMP prior: `a_i = [r_i, nu_r_i]`
- init input: `inp_i = [c_i || a_i]`
- projected hidden: `u_i^0 = W_init * inp_i + b_init`

## 4) 多头掩码注意力块

For each layer `l` and head `k`:

```python
for l in range(n_conv):
    for k in range(n_heads):
        q_i = Wq[k] @ u_i
        k_j = Wk[k] @ u_j
        v_j = Wv[k] @ u_j

        if use_edge_attr:
            e_ij = We[k] @ edge_attr_ij
            score_ij = LeakyReLU(a[k]^T [q_i || k_j || e_ij])
        else:
            score_ij = LeakyReLU(a[k]^T [q_i || k_j])

        score_ij = mask_with_adj(score_ij, adj)   # invalid edges -> -inf
        alpha_ij = softmax_j(score_ij)            # over j in N(i)
        alpha_ij = dropout(alpha_ij, p=attn_dropout)
        m_i_head[k] = sum_j alpha_ij * v_j

    # default head fusion: mean
    m_i = mean_k(m_i_head[k])

    # recurrent update with AMP prior
    s_i = GRUCell(input=[m_i || a_i], hidden=s_i)
    u_i = W_out @ s_i + b_out
```

## 5) 读出到 `x_hat, nu_x`

```python
logits_i = MLP_readout(u_i)       # (B,n,N_Q)
p_i = softmax(logits_i, dim=-1)
x_hat_i = sum_s Q[s] * p_i[s]
nu_x_i  = sum_s (Q[s] - x_hat_i)^2 * p_i[s]
nu_x = clamp(nu_x, min=1e-10)
```

## 6) 实现默认值
- `n_heads = 2`
- head fusion = `mean`
- `mode='gram_triplet'` for edge features
- self-loop enforced in adjacency
- `attn_dropout=0.0` 作为基线；仅在明显过拟合时再启用
