# GAT-OTFS 论文复现实验（cmp_otfs_gat_paper）

本 README 仅面向 `cmp_otfs_gat_paper` 分支中的 GAT-OTFS 复现实验流程，不覆盖旧的 AFDM/AMP-GNN/GAT 通用说明。

## 30 秒快速开始

```bash
py -3 -m pip install -r src/python/requirements.txt
py -3 -m pip install h5py
py -3 src/python/cmp_otfs_gat_paper/train_gat_otfs.py -c src/python/config/gat_otfs_config.yaml
py -3 src/python/cmp_otfs_gat_paper/eval_ber_gat_otfs.py -c src/python/config/gat_otfs_config.yaml
```

如果你当前没有可用 `.mat` 数据，先用 `--synthetic` 跑通流程：

```bash
py -3 src/python/cmp_otfs_gat_paper/train_gat_otfs.py --synthetic
```

## 1. 项目简介

本分支复现论文 **Sparse Graph Attention Network Based Signal Detection for OTFS System** 的核心训练与评估流程。  
主要入口脚本：

- 训练：`src/python/cmp_otfs_gat_paper/train_gat_otfs.py`
- 评估：`src/python/cmp_otfs_gat_paper/eval_ber_gat_otfs.py`
- 默认配置：`src/python/config/gat_otfs_config.yaml`

## 2. 目录结构

```text
src/
  python/
    config/
      gat_otfs_config.yaml
    cmp_otfs_gat_paper/
      train_gat_otfs.py
      eval_ber_gat_otfs.py
      gat_otfs_detector.py
      gat_otfs_module.py
      gat_otfs_init.py
      gat_otfs_loss.py
      checkpoints/                       # 训练后自动创建
      experiment_log_gat_otfs_*.csv      # 训练后自动生成
```

## 3. 环境依赖

基础依赖来自 `src/python/requirements.txt`：

- `numpy>=1.20`
- `scipy>=1.7`
- `torch>=1.10`
- `matplotlib>=3.5`
- `PyYAML>=5.4`

如果使用 `.mat` 数据链路，建议补充安装：

- `h5py`

安装示例：

```bash
py -3 -m pip install -r src/python/requirements.txt
py -3 -m pip install h5py
```

## 4. 数据准备

默认配置期望你提供 `train/val/test/meta` 四类路径：

```yaml
data:
  train_dataset_path: "../../data/snr_14db/afdm_train_snr_14db.mat"
  val_dataset_path: "../../data/snr_14db/afdm_val_snr_14db.mat"
  test_dataset_path: "../../data/snr_14db/afdm_test_snr_10db_16db.mat"
  dataset_meta_path: "../../data/snr_14db/dataset_meta.yaml"
```

约定与注意事项：

- 路径按“配置文件所在目录”解析相对路径。
- `data.dataset_path` 已废弃，代码会报错并提示改用分 split 字段。
- 训练脚本在数据缺失或 metadata 校验失败时，会自动回退到 synthetic 模式并继续执行。

## 5. 配置说明

配置文件：`src/python/config/gat_otfs_config.yaml`。

关键结构：

- `data`：数据路径与 synthetic 回退参数（如 `N_dim_half`）。
- `configs`：可配置多组实验；每组含：
- `name`：配置名（决定 checkpoint 文件名）。
- `model`：`F/F_prime/T/Nh1/Nh2/S/adj_eps/add_self_loop`。
- `train`：`batch_size/val_batch_size/lr/n_epoch/max_grad_norm/scheduler_*` 等。

执行逻辑：

- 若 `configs` 存在，训练脚本会按列表逐项训练。
- 每项配置保存为 `checkpoints/<name>.pt`。

## 6. 训练流程

1) 默认配置训练：

```bash
py -3 src/python/cmp_otfs_gat_paper/train_gat_otfs.py
```

2) 显式指定配置：

```bash
py -3 src/python/cmp_otfs_gat_paper/train_gat_otfs.py -c src/python/config/gat_otfs_config.yaml
```

3) 强制 synthetic 数据：

```bash
py -3 src/python/cmp_otfs_gat_paper/train_gat_otfs.py --synthetic
```

训练时会输出每 epoch 的 `loss/val_loss/val_BER/lr`，并在验证损失变优时更新 checkpoint。

## 7. 评估流程

基础评估命令：

```bash
py -3 src/python/cmp_otfs_gat_paper/eval_ber_gat_otfs.py -c src/python/config/gat_otfs_config.yaml
```

可选参数：

- `--checkpoint-dir`：checkpoint 目录（默认 `src/python/cmp_otfs_gat_paper/checkpoints`）。
- `--batch-size`：评估 batch size（默认 `64`）。

示例：

```bash
py -3 src/python/cmp_otfs_gat_paper/eval_ber_gat_otfs.py ^
  -c src/python/config/gat_otfs_config.yaml ^
  --checkpoint-dir src/python/cmp_otfs_gat_paper/checkpoints ^
  --batch-size 128
```

评估输出包括每个配置按 SNR 分组的 BER 和最终 summary（成功/缺失 checkpoint/失败数量）。

## 8. 输出文件说明

训练输出：

- checkpoint：`src/python/cmp_otfs_gat_paper/checkpoints/<config_name>.pt`
- 日志：`src/python/cmp_otfs_gat_paper/experiment_log_gat_otfs_<timestamp>.csv`

CSV 关键字段：

- `row_type`：`epoch` 或 `summary`
- `val_BER`：每 epoch 验证 BER
- `best_epoch`、`best_val_BER`：最佳结果摘要
- `F/F_prime/T/Nh1/Nh2/l2_weight`：核心配置追踪

## 9. 与现有 AMP-GAT 的差异（精简）

cmp 复现脚本相对仓库中原有 AMP-GAT 流程的主要差异：

- 模型直接接收 `(y, H, sigma2)`，在模型内部构图与特征初始化。
- 主损失为交叉熵（可选 L2 辅助项），而非纯 L2。
- BER 直接从 `logits` 推导，不依赖连续值 `x_hat`。
- 支持 synthetic fallback，便于无 MATLAB 数据时先验证训练链路。

## 10. FAQ

1) 运行时提示 `Config not found: ...`？

- 检查 `-c` 路径是否存在。
- 建议优先使用：`-c src/python/config/gat_otfs_config.yaml`。

2) 报 `Dataset/metadata validation failed`？

- 说明 `.mat` 与 `dataset_meta.yaml` 不匹配、字段缺失或路径错误。
- 训练脚本会自动回退 synthetic；修复真实数据后可重新运行。

3) 评估时提示 `checkpoint not found` 或 `checkpoint/model mismatch`？

- 前者通常是配置名与 checkpoint 文件名不一致。
- 后者通常是当前 `model` 超参与训练该 checkpoint 的超参不一致。

4) 没有 CUDA 可以跑吗？

- 可以。脚本会自动选择 `cpu`（训练和评估都可运行，但更慢）。

5) 没有 AFDM 数据还能验证流程吗？

- 可以，使用 `--synthetic`。
- 或在真实数据模式下，当数据路径/元数据无效时，训练脚本也会自动切换到 synthetic。

6) 本分支是否修改了公共 API 或配置 schema？

- 没有。本次仅文档重写，不改 Python API、配置字段定义或代码行为。
