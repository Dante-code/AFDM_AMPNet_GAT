# 伪代码：AMP-GAT 训练流程

## 1) 配置解析

模型配置新增项：
- `model_type`: `amp_gnn` or `amp_gat`
- `damp`
- `n_heads`
- `attn_dropout`
- `use_edge_attr`
- `edge_attr_mode`

## 2) 模型工厂

```python
def build_model(model_cfg, n_dim):
    if model_cfg.model_type == "amp_gnn":
        return AMPGNNDetector(...)
    if model_cfg.model_type == "amp_gat":
        return AMPGATDetector(
            n_dim=n_dim,
            n_iter=model_cfg.n_iter,
            damp=model_cfg.damp,
            n_heads=model_cfg.n_heads,
            attn_dropout=model_cfg.attn_dropout,
            use_edge_attr=model_cfg.use_edge_attr,
            ...
        )
    raise ValueError("unknown model_type")
```

## 3) Batch 图构建

```python
for x, y, H, sigma2 in loader:
    x, y, H, sigma2 = to_device(...)
    adj = build_adjacency(H, eps=1e-8)

    edge_attr = None
    if model_type == "amp_gat" and use_edge_attr:
        edge_attr = build_edge_attr(H, mode=edge_attr_mode)
```

## 4) 训练循环

```python
model.train()
for x, y, H, sigma2 in train_loader:
    adj, edge_attr = build_graph_inputs(H)
    optimizer.zero_grad()

    if model_type == "amp_gat":
        x_hat = model(y, H, sigma2, adj, edge_attr=edge_attr)
    else:
        x_hat = model(y, H, sigma2, adj)

    loss_l2 = ((x - x_hat)**2).mean()

    if use_ce_loss:
        loss_ce = symbol_ce_loss(x, x_hat_logits_or_probs)
        loss = loss_l2 + ce_weight * loss_ce
    else:
        loss = loss_l2

    loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
```

## 5) 验证循环

```python
model.eval()
with no_grad():
    for x, y, H, sigma2 in val_loader:
        adj, edge_attr = build_graph_inputs(H)
        x_hat = model(...)
        val_loss += l2(x, x_hat)
        val_ber  += compute_ber(x, x_hat, M=1, N=N) * batch_size
```

## 6) 学习率调度与检查点规则
- 调度器默认使用 `train_loss_avg`（与当前脚本兼容）；如有需要可改为 `val_loss_avg`。
- 最优 checkpoint 判据（默认）：`val_loss_avg` 最小。
- 多组配置运行时，模型按配置名后缀保存。

## 7) 记录指标
- `train_loss`, `val_loss`, `val_BER`
- `best_epoch`, `best_val_loss`, `best_val_BER`
- `model_type`, `damp`, `n_heads`, `use_edge_attr`, `attn_dropout`

## 8) 冒烟测试流程（正式训练前）
1. Run `n_epoch=2`, small batch subset.
2. Assert no NaN/Inf in loss or variance tensors.
3. Confirm `val_loss` trend decreases vs random initialization.
4. 确认 AMP-GAT 路径在有/无边特征两种模式都可运行。
