# AFDM_AMPNet_GAT

AFDM 数据集由 MATLAB 生成，Python 读取数据集进行 AMP-GNN / AMP-GAT 的训练与评估。

## 数据组织

推荐按训练信噪比分目录存放：

```text
src/data/
  snr_14db/
    afdm_train_snr_14db.mat
    afdm_val_snr_14db.mat
    afdm_test_snr_10db_16db.mat
    dataset_meta.yaml
```

## 配置字段（当前版本）

Python 配置文件位于 `src/python/config/*.yaml`，`data` 段使用以下字段：

```yaml
data:
  train_dataset_path: "../../data/snr_14db/afdm_train_snr_14db.mat"
  val_dataset_path: "../../data/snr_14db/afdm_val_snr_14db.mat"
  test_dataset_path: "../../data/snr_14db/afdm_test_snr_10db_16db.mat"
  dataset_meta_path: "../../data/snr_14db/dataset_meta.yaml"
  SNR_train: 14
  l_max: 2
  k_max: 3
```

说明：
- 路径相对于“配置文件所在目录”解析。
- 旧字段 `data.dataset_path` 已废弃，当前代码会报错并提示改用新字段。

## MATLAB 生成数据

文件：`src/generate_afdm_dataset.m`

- `split_mode`：`"train"` / `"val"` / `"test"`，单次只生成一个 split。
- `SNR_dB`：训练 SNR，同时用于自动输出目录 `./data/snr_<SNR_dB>db/`。
- 每次运行会在对应目录更新 `dataset_meta.yaml`。

## Python 训练与评估

训练：

```bash
py -3 src/python/train_afdm.py -c src/python/config/gat_snr14db_config.yaml
```

评估：

```bash
py -3 src/python/eval_ber_afdm.py -c src/python/config/gat_snr14db_config.yaml
```
