# 伪代码：AMPGATDetector

## 类接口

```python
class AMPGATDetector(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_iter: int = 4,
        damp: float = 0.7,
        n_heads: int = 2,
        attn_dropout: float = 0.0,
        use_edge_attr: bool = True,
        edge_attr_dim: int = 3,
        **gat_kwargs
    ):
        ...

    def forward(
        self,
        y: Tensor,         # (B, n)
        H: Tensor,         # (B, n, n)
        sigma2: Tensor,    # (B,)
        adj: Tensor,       # (B, n, n)
        edge_attr: Tensor | None = None  # (B, n, n, d_e)
    ) -> Tensor:           # returns x_hat: (B, n)
        ...
```

## 张量约定
- `B`: batch size
- `n`: flattened real symbol dimension (`2MN`)
- `x_hat, nu_x, r, nu_r, z, nu_z`: all `(B, n)`
- `adj`: binary/float mask `(B, n, n)`
- `edge_attr`: `(B, n, n, d_e)` when enabled

## Forward 伪代码

```python
def forward(y, H, sigma2, adj, edge_attr=None):
    eps = 1e-10
    B, n = y.shape

    # init states
    x_hat = zeros(B, n)
    nu_x  = full((B, n), 0.5)
    z     = y.clone()
    nu_z  = bmm(H**2, nu_x.unsqueeze(-1)).squeeze(-1) + eps

    for t in range(n_iter):
        # 1) AMP linear step
        z, nu_z, r, nu_r = amp_linear_step(
            y=y, H=H, x_hat=x_hat, nu_x=nu_x, z_prev=z, nu_z_prev=nu_z, sigma2=sigma2, eps=eps
        )

        # 2) GAT graph step
        x_new, nu_new = gat(
            y=y, H=H, r=r, nu_r=nu_r, adj=adj, edge_attr=edge_attr
        )

        # 3) Damping (optional, controlled by damp in [0,1])
        if damp < 1.0:
            x_hat = damp * x_new + (1.0 - damp) * x_hat
            nu_x  = damp * nu_new + (1.0 - damp) * nu_x
        else:
            x_hat = x_new
            nu_x  = nu_new

        nu_x = clamp(nu_x, min=eps)

    return x_hat
```

## 状态变量生命周期
- `x_hat, nu_x`：跨迭代保留，作为后验估计状态。
- `z, nu_z`：AMP 隐状态，跨迭代保留。
- `r, nu_r`：当前迭代 AMP 的瞬时输出。
- `x_new, nu_new`：当前迭代 GAT 输出，经过阻尼融合后写回状态。

## 失败处理规则
- 所有方差类变量裁剪到 `>= eps`。
- 若 `adj` 某行没有邻居，强制自环或回退为恒等消息。
- 若 `use_edge_attr=True` 但 `edge_attr is None`，在线从 Gram 矩阵构建回退边特征。

## 最小单元检查
1. 第一轮前检查所有输入张量 shape。
2. 检查 `x_hat`, `nu_x`, `nu_r` 的有限性（`isfinite`）。
3. 检查 `nu_*` 不出现负值。
