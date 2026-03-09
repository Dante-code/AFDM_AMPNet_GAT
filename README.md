已按你给的清单实施完成，当前状态已经对齐 `src/data/snr_xxdb` 组织方式。

**已修改**
1. 修正 `gat_snr14db_config.yaml` 路径与 `SNR_train`
- [gat_snr14db_config.yaml](E:\code\AFDM_AMPNet_GAT\src\python\config\gat_snr14db_config.yaml:4)
- train/val 文件名改为 `afdm_train_snr_14db.mat` / `afdm_val_snr_14db.mat`
- `dataset_meta_path` 改为 `../../data/snr_14db/dataset_meta.yaml`
- `SNR_train` 改为 `14`

2. 同步更新通用配置，避免误用旧路径
- [train_gat_config.yaml](E:\code\AFDM_AMPNet_GAT\src\python\config\train_gat_config.yaml:4)
- [train_gnn_config.yaml](E:\code\AFDM_AMPNet_GAT\src\python\config\train_gnn_config.yaml:4)
- 两个文件都改为指向 `../../data/snr_14db/...`，并将 `SNR_train` 改为 `14`

3. MATLAB 默认输出目录按 `SNR_dB` 自动分桶
- [generate_afdm_dataset.m](E:\code\AFDM_AMPNet_GAT\src\generate_afdm_dataset.m:11)
- 新增 `output_root`，并自动生成 `output_dir = ./data/snr_<SNR_dB>db`
- metadata 继续写到该目录下（`dataset_meta.yaml`）
- `test_mat_name` 保持 `afdm_test_snr_10db_16db`，与当前测试集文件命名一致

4. Python 增强 metadata 一致性校验
- [dataset_afdm.py](E:\code\AFDM_AMPNet_GAT\src\python\dataset_afdm.py:145)
- 新增校验：
  - `l_max/k_max` 一致性
  - `snr_train_db`（train/val）一致性
  - metadata 的 `splits.<split>.file` 与 config 指向文件 basename 必须一致
- 同时支持读取并携带 `SNR_dB` 字段用于上述校验

5. 新增并对齐 `snr_14db` 目录下 metadata
- [dataset_meta.yaml](E:\code\AFDM_AMPNet_GAT\src\data\snr_14db\dataset_meta.yaml:1)
- `splits.test.file` 已与实际文件名统一为 `afdm_test_snr_10db_16db.mat`

**验证结果**
- 已确认 `src/data/snr_14db/` 下存在：
  - `afdm_train_snr_14db.mat`
  - `afdm_val_snr_14db.mat`
  - `afdm_test_snr_10db_16db.mat`
  - `dataset_meta.yaml`
- Python 关键脚本已做语法解析检查通过（`ast.parse`）。

如果你愿意，我下一步可以再帮你跑一轮“训练入口 + 评估入口”的实际启动检查（只跑到数据加载与配置校验阶段）。