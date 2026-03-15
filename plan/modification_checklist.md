# 2026/3/15 AMP-GNN 工程修改步骤清单

## 修改总览

共分 3 个阶段，按顺序执行，每个阶段完成后做一次冒烟测试再进入下一阶段。

---

## 阶段 1：Python 数据管道适配（让新数据能被正确加载）

### 1.1 扩展 `dataset_afdm.py` — 加载 loc_main

**文件**：`src/python/dataset_afdm.py`

**修改内容**：

在 `_split_keys` 函数中增加 loc_main 的 key 映射：
```python
def _split_keys(split: str) -> dict[str, str]:
    # ... 现有 keys ...
    keys["loc_main"] = f"loc_main_{split}_arr"
    return keys
```

在 `load_afdm_split_mat` 函数中，加载 loc_main 数据（需要兼容旧数据集不含此字段的情况）：
```python
# 在 out dict 构建之后
loc_main_key = keys["loc_main"]
if loc_main_key in data:
    out[f"loc_main_{split}"] = data[loc_main_key]  # shape: (n_samples, P)
# h5 分支也做同样处理
```

在 `_extract_common_from_mat_dict` 和 `_extract_common_from_h5` 中，把 `"kv"` 加入提取列表（已存在于你的代码中，确认即可）。

### 1.2 扩展 `afdm_utils.py` — 构建实值 IDI mask

**文件**：`src/python/afdm_utils.py`

**新增函数**：
```python
def build_idi_mask_from_loc_main(H_real: np.ndarray, loc_main: np.ndarray, 
                                  N: int, P: int, kv: int) -> np.ndarray:
    """
    根据 loc_main (P,) 和 kv 构建实值展开后的 IDI 掩码。
    
    H_real: (2N, 2N) 实值信道矩阵
    loc_main: (P,) 每条路径的主项列偏移（0 基，复数域）
    N: 块大小
    kv: 扩展半宽
    
    返回: mask_idi (2N, 2N) bool，True 表示该位置属于扩展项（IDI）
    """
    n = 2 * N
    # 先在复数域 (N, N) 构建 main_mask
    main_mask_complex = np.zeros((N, N), dtype=bool)
    for p_row in range(N):
        for path_idx in range(P):
            q_main = (p_row + int(loc_main[path_idx])) % N
            main_mask_complex[p_row, q_main] = True
    
    # 扩展到实值域 (2N, 2N)
    # 实值矩阵结构: [[Re(H), -Im(H)], [Im(H), Re(H)]]
    # main_mask 的实值展开与 H 相同
    main_mask_real = np.zeros((n, n), dtype=bool)
    main_mask_real[:N, :N] = main_mask_complex      # Re-Re 块
    main_mask_real[:N, N:] = main_mask_complex       # Re-Im 块
    main_mask_real[N:, :N] = main_mask_complex       # Im-Re 块
    main_mask_real[N:, N:] = main_mask_complex       # Im-Im 块
    
    # 非零且非主项的位置就是 IDI
    nonzero_mask = np.abs(H_real) > 1e-12
    mask_idi = nonzero_mask & (~main_mask_real)
    
    return mask_idi
```

**修改 `prepare_sample`**：增加可选的 loc_main/kv 参数，当提供时返回 mask_idi：
```python
def prepare_sample(x_vec, y_vec, H_eff, sigma2, 
                   loc_main=None, N=None, P=None, kv=0):
    # ... 现有代码 ...
    
    mask_idi = None
    if loc_main is not None and kv > 0:
        mask_idi = build_idi_mask_from_loc_main(H, loc_main, N, P, kv)
    
    return x, y, H, sigma2_real, I_list, L_list, mask_idi
```

### 1.3 修改 `train_afdm.py` — Dataset 和 collate_fn 传递 mask_idi

**文件**：`src/python/train_afdm.py`

**修改 `AFDMDataset.__getitem__`**：
```python
def __getitem__(self, idx):
    split = self.split
    x_vec = self.raw[f"x_daf_{split}"][idx]
    y_vec = self.raw[f"y_daf_{split}"][idx]
    H_eff = self.raw[f"H_eff_{split}"][idx]
    sigma2 = self.raw[f"sigma2_{split}"][idx]
    
    # 新增：读取 loc_main（兼容旧数据集）
    loc_main = None
    kv = int(self.raw.get("kv", 0))
    loc_main_key = f"loc_main_{split}"
    if loc_main_key in self.raw and kv > 0:
        loc_main = self.raw[loc_main_key][idx]
    
    N = self.raw["N"]
    P = self.raw.get("P", 4)
    x, y, H, sigma2_r, _, _, mask_idi = afdm_utils.prepare_sample(
        x_vec, y_vec, H_eff, sigma2,
        loc_main=loc_main, N=N, P=P, kv=kv
    )
    return x, y, H, sigma2_r, mask_idi  # 多返回一个 mask_idi
```

**修改 `collate_fn`**：
```python
def collate_fn(batch):
    x = np.stack([b[0] for b in batch])
    y = np.stack([b[1] for b in batch])
    H = np.stack([b[2] for b in batch])
    sigma2 = np.array([b[3] for b in batch])
    
    # mask_idi 可能为 None（旧数据集或 kv=0）
    if batch[0][4] is not None:
        mask_idi = np.stack([b[4] for b in batch])
        mask_idi_t = torch.tensor(mask_idi, dtype=torch.bool)
    else:
        mask_idi_t = None
    
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(H, dtype=torch.float32),
        torch.tensor(sigma2, dtype=torch.float32),
        mask_idi_t,
    )
```

### 1.4 冒烟测试

- 用新数据集跑 `AFDMDataset` 的 `__getitem__`，确认返回的 mask_idi shape 为 (2N, 2N)
- 确认 mask_idi 中 True 的数量 ≈ 每行 P×2kv 个（扩展项），False 的剩余中主项约 P 个
- 在旧数据集（kv=0）上跑同样的代码，确认 mask_idi 为 None，不影响现有功能

---

## 阶段 2：新建 IDI 近似模块

### 2.1 创建 `src/python/idi_approx.py`

**文件**：新建 `src/python/idi_approx.py`

```python
"""
IDI (Inter-Doppler Interference) approximation module.
Implements the normalization scheme from AMP-GNN paper (Zhuang et al. 2024),
adapted for AFDM fractional Doppler scenarios.
"""
import torch


def compute_idi_stats(
    H: torch.Tensor,           # (B, n, n) 实值信道矩阵
    mask_idi: torch.Tensor,    # (B, n, n) bool, True = IDI 位置
    x_hat: torch.Tensor,       # (B, n) 上一轮后验均值
    nu_x: torch.Tensor,        # (B, n) 上一轮后验方差
    sigma2: torch.Tensor,      # (B,) 实值噪声方差 σ²_n/2
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算 IDI + 噪声的高斯近似统计量。
    
    对应论文公式 (12a)-(12b)：
      μ_ζ_j = Σ_{i ∈ IDI(j)} h_ji * x̂_i
      σ²_ζ_j = Σ_{i ∈ IDI(j)} h²_ji * ν̂_{x,i} + σ²_n/2
    
    返回: (mu_zeta, sigma2_zeta), 都是 (B, n)
    """
    # 把非 IDI 位置的 H 置零，得到 H_idi
    H_idi = H * mask_idi.float()  # (B, n, n)
    
    # μ_ζ = H_idi @ x̂
    mu_zeta = torch.bmm(H_idi, x_hat.unsqueeze(-1)).squeeze(-1)  # (B, n)
    
    # σ²_ζ = (H_idi²) @ ν̂_x + σ²/2
    s2 = sigma2.view(-1, 1) + eps  # (B, 1)
    sigma2_zeta = torch.bmm(
        H_idi ** 2, nu_x.unsqueeze(-1)
    ).squeeze(-1) + s2  # (B, n)
    
    sigma2_zeta = sigma2_zeta.clamp(min=eps)
    
    return mu_zeta, sigma2_zeta


def normalize_signal_and_channel(
    y: torch.Tensor,            # (B, n)
    H: torch.Tensor,            # (B, n, n)
    mask_idi: torch.Tensor,     # (B, n, n) bool
    mu_zeta: torch.Tensor,      # (B, n)
    sigma2_zeta: torch.Tensor,  # (B, n)
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    归一化接收信号和信道矩阵，吸收 IDI 为等效白噪声。
    
    对应论文公式 (13)：
      ỹ_j = (y_j - μ_ζ_j) / σ_ζ_j
      H̃_ji = h_ji / σ_ζ_j    （仅保留主路径项，IDI 列置零）
    
    返回: (y_tilde, H_tilde)
    """
    sigma_zeta = torch.sqrt(sigma2_zeta).clamp(min=eps)  # (B, n)
    
    # ỹ = (y - μ_ζ) / σ_ζ
    y_tilde = (y - mu_zeta) / sigma_zeta  # (B, n)
    
    # H̃ = H_main / σ_ζ （把 IDI 列置零，只保留主路径列）
    mask_main = ~mask_idi  # (B, n, n) 主路径 + 零元素的位置
    H_main = H * mask_main.float()
    
    # 逐行除以 σ_ζ_j：H̃[j,i] = H_main[j,i] / σ_ζ[j]
    H_tilde = H_main / sigma_zeta.unsqueeze(-1)  # (B, n, n)
    
    return y_tilde, H_tilde
```

### 2.2 单元测试 `idi_approx.py`

写一个简单的测试脚本验证：
- 当 mask_idi 全 False（无 IDI，即整数多普勒）时，mu_zeta=0，H_tilde ≈ H/σ_n，行为退化到无 IDI
- 当 mask_idi 有值时，H_tilde 的非零元素数量应 < H 的非零元素数量
- sigma2_zeta 全部 > 0，无 NaN/Inf
- y_tilde 和 H_tilde 的 shape 正确

---

## 阶段 3：修改检测器核心迭代逻辑

### 3.1 修改 `amp_gnn_detector.py` — 加入 IDI 近似

**文件**：`src/python/amp_gnn_detector.py`

**修改 `AMPGNNDetector.__init__`**：增加 `use_idi_approx` 参数：
```python
def __init__(self, n_dim: int, n_iter: int = 3, 
             use_idi_approx: bool = False, **gnn_kwargs):
    super().__init__()
    self.n_dim = n_dim
    self.n_iter = n_iter
    self.use_idi_approx = use_idi_approx
    self.gnn = gnn_module.GNNModule(n_dim, **gnn_kwargs)
```

**修改 `AMPGNNDetector.forward`**：

```python
def forward(self, y, H, sigma2, adj, mask_idi=None):
    B, n = y.shape
    device = y.device
    eps = 1e-10

    # 初始化（修复：nu_x 初始化为 0.5，与论文一致）
    x_hat = torch.zeros(B, n, device=device)
    nu_x = torch.full((B, n), 0.5, device=device)  # ← 从 1.0 改为 0.5
    z = y.clone()
    nu_z = torch.bmm((H ** 2), nu_x.unsqueeze(-1)).squeeze(-1) + eps

    for t in range(self.n_iter):
        # ---- IDI 近似：归一化 ----
        if self.use_idi_approx and mask_idi is not None:
            from idi_approx import compute_idi_stats, normalize_signal_and_channel
            mu_z, sigma2_z = compute_idi_stats(H, mask_idi, x_hat, nu_x, sigma2, eps)
            y_t, H_t = normalize_signal_and_channel(y, H, mask_idi, mu_z, sigma2_z, eps)
            # 归一化后等效噪声方差 = 0.5（实值分解后的 ζ̃ ~ N(0, 1/2)）
            sigma2_t = torch.full_like(sigma2, 0.5)
            # 动态重建邻接矩阵
            adj_t = build_adjacency(H_t)
        else:
            y_t, H_t, sigma2_t, adj_t = y, H, sigma2, adj

        # ---- AMP 步 ----
        z, nu_z, r, nu_r = amp_linear.amp_linear_step(
            y_t, H_t, x_hat, nu_x, z, nu_z, sigma2_t, eps
        )

        # ---- GNN 步（使用归一化后的 y_t, H_t, adj_t）----
        x_hat, nu_x = self.gnn(y_t, H_t, r, nu_r, adj_t)

    return x_hat
```

**关键变化说明**：
- AMP 和 GNN 都使用归一化后的 (y_t, H_t, sigma2_t)
- 邻接矩阵 adj_t 每轮从 H̃ 重建（动态图）
- 当 use_idi_approx=False 或 mask_idi=None 时，退化为原来的行为（向后兼容）
- nu_x 初始值从 1.0 修正为 0.5

**注意 import**：需要在文件顶部加 `from graph_features import build_adjacency`（或在现有 import 中确认）。idi_approx 的 import 放在分支内部是为了避免旧代码路径的依赖问题，也可以提到顶部。

### 3.2 同步修改 `amp_gat_detector.py`

**文件**：`src/python/amp_gat_detector.py`

做与 3.1 完全对称的改动：
- `__init__` 增加 `use_idi_approx` 参数
- `forward` 中在 AMP 步之前插入 IDI 归一化
- adj 和 edge_attr 每轮从 H̃ 重建
- nu_x 初始值改为 0.5

### 3.3 修改 `gnn_module.py` — 边特征维度修正（可选但推荐）

**文件**：`src/python/gnn_module.py`

**当前问题**：消息 `[u_i, u_j, e_ij]` 中 e_ij 被 expand 到 n_u 维，MLP 输入 3*n_u。论文中 e_ij 是标量，输入应为 2*n_u+1。

**修改 `__init__`**：
```python
# 消息聚合 MLP 输入维度：2*n_u + 1（两个节点特征 + 一个标量边权）
self.msg_mlp = nn.Sequential(
    nn.Linear(2 * n_u + 1, n_mlp_hidden),
    nn.ReLU(),
    nn.Linear(n_mlp_hidden, n_u),
)
```

**修改 `forward` 中消息构造**：
```python
# 不再 expand e_ij 到 n_u 维，直接用标量
u_i = u.unsqueeze(2).expand(-1, -1, n, -1)   # (B, n, n, n_u)
u_j = u.unsqueeze(1).expand(-1, n, -1, -1)   # (B, n, n, n_u)
e_ij = G.unsqueeze(-1)                        # (B, n, n, 1) ← 标量边权
msg_j2i = torch.cat([u_i, u_j, e_ij], dim=-1)  # (B, n, n, 2*n_u+1)
```

### 3.4 修改 `train_afdm.py` — forward_model 传递 mask_idi

**文件**：`src/python/train_afdm.py`

**修改 `forward_model`**：
```python
def forward_model(model, model_cfg: dict, y, H, sigma2, mask_idi=None):
    model_type = model_cfg.get("model_type", "amp_gnn")
    adj = build_adjacency(H)  # 初始 adj（IDI 模式下会在内部被覆盖）
    if model_type == "amp_gat":
        edge_attr = None
        if bool(model_cfg.get("use_edge_attr", True)):
            edge_attr = build_edge_attr(H, mode=model_cfg.get("edge_attr_mode", "gram_triplet"))
        return model(y, H, sigma2, adj, edge_attr=edge_attr, mask_idi=mask_idi)
    return model(y, H, sigma2, adj, mask_idi=mask_idi)
```

**修改 `build_model`**：
```python
if model_type == "amp_gnn":
    model = AMPGNNDetector(
        n_dim, n_iter=n_iter,
        use_idi_approx=bool(model_cfg.get("use_idi_approx", False)),
        n_u=n_u, n_h=n_h, n_conv=n_conv, n_mlp_hidden=n_mlp_hidden
    )
```

**修改训练/验证循环**：在解包 batch 时多取 mask_idi：
```python
for batch in train_loader:
    x, y, H, sigma2, mask_idi = batch  # 多解包一个
    x, y, H, sigma2 = x.to(device), y.to(device), H.to(device), sigma2.to(device)
    if mask_idi is not None:
        mask_idi = mask_idi.to(device)
    
    x_hat = forward_model(model, model_cfg, y, H, sigma2, mask_idi=mask_idi)
    # ... loss, backward, step ...
```

### 3.5 修改 `eval_ber_afdm.py` — 评估路径同步

**文件**：`src/python/eval_ber_afdm.py`

做与 train 类似的改动：
- `build_model` 增加 `use_idi_approx` 传递
- `forward_model` 增加 `mask_idi` 参数
- 评估循环中构建 mask_idi 并传入

### 3.6 更新配置文件

**文件**：所有 `src/python/config/*.yaml`

在 model 配置块中增加 IDI 相关开关：
```yaml
configs:
  - name: "amp_gnn_idi"
    model:
      model_type: "amp_gnn"
      n_iter: 6
      n_u: 10
      n_h: 14
      n_conv: 4
      n_mlp_hidden: 22
      use_idi_approx: true    # ← 新增
      # ... 其余不变 ...
  
  - name: "amp_gnn_no_idi"
    model:
      model_type: "amp_gnn"
      use_idi_approx: false   # ← 对照组
      # ... 其余不变 ...
```

---

## 阶段 4：验证与对比实验

### 4.1 退化验证（最重要）

用**整数多普勒数据集**（kv=0）跑 IDI 近似版本：
- 此时 mask_idi 全 False（或 None）
- 检测器应退化为原来的行为
- BER 应与修改前完全一致

### 4.2 分数多普勒基线

用**分数多普勒数据集**（kv=1）跑 `use_idi_approx: false`：
- 此时 GNN 直接面对密集 H（每行 P×3 个非零元素）
- 记录 BER 和训练时间作为基线

### 4.3 IDI 近似效果验证

用同一数据集跑 `use_idi_approx: true`：
- 预期 BER 接近 4.2 的基线（近似几乎无损）
- 预期 per-epoch 时间显著下降（图变稀疏）

### 4.4 AMP-GAT + IDI 近似

重复 4.2/4.3 对 AMP-GAT 做同样实验。

### 4.5 记录的指标

每次运行记录：
- config_name, use_idi_approx, kv
- best_val_loss, best_val_BER
- per_epoch_time (秒)
- 非零邻居数（nnz per row of adj）

---

## 各文件修改一览表

| 文件 | 阶段 | 修改类型 | 说明 |
|------|------|----------|------|
| `buildHeff_DAF.m` | 0 | Bug fix | alpha = kp |
| `generateChannel.m` | 0 | Bug fix | 功率模型 sqrt(1/(2P)) |
| `generate_afdm_dataset.m` | 0 | 重新生成 | 修完后重跑 |
| `dataset_afdm.py` | 1 | 扩展 | 加载 loc_main |
| `afdm_utils.py` | 1 | 扩展 | build_idi_mask + prepare_sample |
| `train_afdm.py` | 1+3 | 修改 | Dataset, collate_fn, forward_model, 训练循环 |
| `idi_approx.py` | 2 | **新建** | compute_idi_stats + normalize |
| `amp_gnn_detector.py` | 3 | 修改 | IDI 归一化 + nu_x=0.5 + 动态 adj |
| `amp_gat_detector.py` | 3 | 修改 | 同上 |
| `gnn_module.py` | 3 | 修改 | 边特征维度 3n_u → 2n_u+1（可选） |
| `eval_ber_afdm.py` | 3 | 修改 | 同步 mask_idi 传递 |
| `config/*.yaml` | 3 | 修改 | 增加 use_idi_approx 开关 |
