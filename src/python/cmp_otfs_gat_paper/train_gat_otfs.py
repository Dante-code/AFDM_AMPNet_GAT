"""
GAT-OTFS 训练脚本 -- 复现论文 "Sparse Graph Attention Network Based Signal Detection for OTFS System"

用法:
    py -3 src/python/cmp_otfs_gat_paper/train_gat_otfs.py -c src/python/config/gat_otfs_config.yaml

特性:
    - 复用现有 AFDM 数据加载流程 (dataset_afdm + afdm_utils)
    - 交叉熵损失 (论文公式 33)
    - 可选 L2 辅助损失
    - CSV 实验日志
    - checkpoint 保存/恢复
    - 学习率调度 (ReduceLROnPlateau)
    - 梯度裁剪

与 train_afdm.py 的区别:
    1. 模型为 GATOTFSDetector (无 AMP 外层循环)
    2. 损失函数为交叉熵 (非 L2)
    3. 模型直接接收 (y, H, sigma2), 不构建 adj/edge_attr 在外面
    4. BER 从 logits 计算 (非从连续 x_hat)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

# ---- 路径设置 ----
_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
sys.path.insert(0, _python_root)

# 导入现有数据工具 (如果在现有仓库中运行)
# 如果独立运行, 需要确保 afdm_utils.py 和 dataset_afdm.py 在 sys.path 中
try:
    import afdm_utils
    import dataset_afdm
    HAS_AFDM_DATA = True
except ImportError:
    HAS_AFDM_DATA = False
    print("Warning: import failed for afdm_utils/dataset_afdm; synthetic fallback is enabled.")

from gat_otfs_detector import GATOTFSDetector
from gat_otfs_loss import (
    GATOTFSCELoss,
    compute_ber_from_logits,
    compute_ber_from_symbols,
    symbols_to_labels,
    N_Q,
)


# ============================================================
# 1) 数据集
# ============================================================
class AFDMDataset(Dataset):
    """复用现有 AFDM 数据格式."""

    def __init__(self, raw_data: dict, split: str):
        self.raw = raw_data
        self.split = split
        self.n = raw_data[f"n_{split}"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        split = self.split
        x_vec = self.raw[f"x_daf_{split}"][idx]
        y_vec = self.raw[f"y_daf_{split}"][idx]
        H_eff = self.raw[f"H_eff_{split}"][idx]
        sigma2 = self.raw[f"sigma2_{split}"][idx]
        x, y, H, sigma2_r, _, _ = afdm_utils.prepare_sample(x_vec, y_vec, H_eff, sigma2)
        return x, y, H, sigma2_r


class SyntheticDataset(Dataset):
    """合成数据集, 用于无 MATLAB 数据时的冒烟测试."""

    def __init__(self, n_samples: int, n_dim: int, snr_db: float = 14.0):
        self.n_samples = n_samples
        self.n_dim = n_dim
        self.snr_db = snr_db
        q = 1.0 / np.sqrt(2)

        # 预生成所有数据
        self.x_data = []
        self.y_data = []
        self.H_data = []
        self.sigma2_data = []

        snr_linear = 10 ** (snr_db / 10)
        sigma2 = 1.0 / snr_linear

        for _ in range(n_samples):
            # QPSK 符号
            x = np.random.choice([-q, q], size=n_dim).astype(np.float64)
            # 稀疏信道 (模拟 OTFS 结构)
            H = np.random.randn(n_dim, n_dim).astype(np.float64) * 0.1
            # 使 H 稀疏 (保留约 10% 非零)
            mask = np.random.rand(n_dim, n_dim) > 0.9
            np.fill_diagonal(mask, True)
            H = H * mask
            # 接收信号
            noise = np.random.randn(n_dim).astype(np.float64) * np.sqrt(sigma2 / 2)
            y = H @ x + noise

            self.x_data.append(x)
            self.y_data.append(y)
            self.H_data.append(H)
            self.sigma2_data.append(sigma2)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.H_data[idx], self.sigma2_data[idx]


def collate_fn(batch):
    """将 batch 中的 numpy 数据转为 tensor."""
    x = np.stack([b[0] for b in batch])
    y = np.stack([b[1] for b in batch])
    H = np.stack([b[2] for b in batch])
    sigma2 = np.array([b[3] for b in batch])
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(H, dtype=torch.float32),
        torch.tensor(sigma2, dtype=torch.float32),
    )


# ============================================================
# 2) 配置解析
# ============================================================
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_list(config: dict) -> list:
    if "configs" in config:
        return config["configs"]
    return [{"name": "single", "model": config["model"], "train": config["train"]}]


def build_detector(model_cfg: dict) -> GATOTFSDetector:
    """从配置构建 GAT-OTFS 检测器."""
    return GATOTFSDetector(
        F=int(model_cfg.get("F", 8)),
        F_prime=int(model_cfg.get("F_prime", 16)),
        T=int(model_cfg.get("T", 10)),
        Nh1=int(model_cfg.get("Nh1", 64)),
        Nh2=int(model_cfg.get("Nh2", 32)),
        S=int(model_cfg.get("S", N_Q)),
        adj_eps=float(model_cfg.get("adj_eps", 1e-8)),
        add_self_loop=bool(model_cfg.get("add_self_loop", True)),
    )


# ============================================================
# 3) 单次配置训练
# ============================================================
def run_one_config(
    cfg_entry: dict,
    raw: dict | None,
    N: int,
    device: torch.device,
    csv_path: str,
    run_id: str,
    csv_fields: list[str],
    write_header: bool,
):
    cfg_name = cfg_entry.get("name", "unnamed")
    model_cfg = cfg_entry["model"]
    train_cfg = cfg_entry["train"]

    # ---- 超参 ----
    batch_size = int(train_cfg.get("batch_size", 32))
    val_batch_size = int(train_cfg.get("val_batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    n_epoch = int(train_cfg.get("n_epoch", 200))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    scheduler_factor = float(train_cfg.get("scheduler_factor", 0.5))
    scheduler_patience = int(train_cfg.get("scheduler_patience", 5))
    l2_weight = float(train_cfg.get("l2_weight", 0.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    n_dim = 2 * N

    # ---- 数据 ----
    if raw is not None:
        train_loader = DataLoader(
            AFDMDataset(raw, "train"),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            AFDMDataset(raw, "val"),
            batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn,
        )
    else:
        # 合成数据 fallback
        n_train = int(train_cfg.get("n_train_synthetic", 1024))
        n_val = int(train_cfg.get("n_val_synthetic", 256))
        train_loader = DataLoader(
            SyntheticDataset(n_train, n_dim),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            SyntheticDataset(n_val, n_dim),
            batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn,
        )

    # ---- 模型 ----
    model = build_detector(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg_name}] 模型参数量: {n_params:,}, 设备: {device}")

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience,
    )
    loss_fn = GATOTFSCELoss(l2_weight=l2_weight, label_smoothing=label_smoothing).to(device)

    # ---- Checkpoint 路径 ----
    save_dir = os.path.join(_script_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{cfg_name}.pt")

    best_val_loss = float("inf")
    best_val_ber = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, n_epoch + 1):
        # ========== 训练 ==========
        model.train()
        total_loss = 0.0
        n_train_batches = 0

        for x, y, H, sigma2 in train_loader:
            x, y, H, sigma2 = x.to(device), y.to(device), H.to(device), sigma2.to(device)
            optimizer.zero_grad()

            logits = model(y, H, sigma2)            # (B, n, S)
            loss, _ = loss_fn(logits, x)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_train_batches += 1

        train_loss_avg = total_loss / max(1, n_train_batches)
        scheduler.step(train_loss_avg)

        # ========== 验证 ==========
        model.eval()
        val_loss_sum = 0.0
        val_ber_sum = 0.0
        n_val_batches = 0
        n_val_samples = 0

        with torch.no_grad():
            for x, y, H, sigma2 in val_loader:
                x, y, H, sigma2 = x.to(device), y.to(device), H.to(device), sigma2.to(device)

                logits = model(y, H, sigma2)
                loss_val, _ = loss_fn(logits, x)
                val_loss_sum += loss_val.item()

                ber_batch = compute_ber_from_logits(logits, x)
                val_ber_sum += ber_batch * len(x)
                n_val_samples += len(x)
                n_val_batches += 1

        val_loss_avg = val_loss_sum / max(1, n_val_batches)
        val_ber_avg = val_ber_sum / max(1, n_val_samples)

        # ========== Checkpoint ==========
        is_best = 0
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_ber = val_ber_avg
            best_epoch = epoch
            is_best = 1
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss_avg,
                "val_ber": val_ber_avg,
                "model_cfg": model_cfg,
                "train_cfg": train_cfg,
            }, save_path)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # ========== 日志 ==========
        row = {k: "" for k in csv_fields}
        row.update({
            "run_id": run_id,
            "config_name": cfg_name,
            "row_type": "epoch",
            "epoch": epoch,
            "n_epoch": n_epoch,
            "train_loss": f"{train_loss_avg:.6f}",
            "val_loss": f"{val_loss_avg:.6f}",
            "val_BER": f"{val_ber_avg:.6e}",
            "is_best": is_best,
            "elapsed_sec": f"{elapsed:.1f}",
            "lr": f"{current_lr:.2e}",
            "F": model_cfg.get("F", 8),
            "F_prime": model_cfg.get("F_prime", 16),
            "T": model_cfg.get("T", 10),
            "Nh1": model_cfg.get("Nh1", 64),
            "Nh2": model_cfg.get("Nh2", 32),
            "batch_size": batch_size,
            "device": str(device),
            "model_type": "gat_otfs",
            "l2_weight": l2_weight,
        })
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        # 控制台输出
        best_mark = " *" if is_best else ""
        print(
            f"[{cfg_name}] epoch {epoch}/{n_epoch} "
            f"loss={train_loss_avg:.4f} val_loss={val_loss_avg:.4f} "
            f"val_BER={val_ber_avg:.4e} lr={current_lr:.2e}{best_mark}"
        )

    # ========== 训练结束 summary ==========
    total_time = time.time() - t0
    summary = {k: "" for k in csv_fields}
    summary.update({
        "run_id": run_id,
        "config_name": cfg_name,
        "row_type": "summary",
        "epoch": 0,
        "n_epoch": n_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": f"{best_val_loss:.6f}",
        "best_val_BER": f"{best_val_ber:.6e}",
        "final_train_loss": f"{train_loss_avg:.6f}",
        "final_val_loss": f"{val_loss_avg:.6f}",
        "final_val_BER": f"{val_ber_avg:.6e}",
        "total_time_sec": f"{total_time:.1f}",
        "N": N,
        "F": model_cfg.get("F", 8),
        "F_prime": model_cfg.get("F_prime", 16),
        "T": model_cfg.get("T", 10),
        "Nh1": model_cfg.get("Nh1", 64),
        "Nh2": model_cfg.get("Nh2", 32),
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "model_type": "gat_otfs",
        "l2_weight": l2_weight,
    })
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writerow(summary)

    print(f"\n[{cfg_name}] 训练完成: best_epoch={best_epoch}, "
          f"best_val_loss={best_val_loss:.4f}, best_val_BER={best_val_ber:.4e}, "
          f"总时间={total_time:.1f}s")
    print(f"  checkpoint: {save_path}")

    return write_header


# ============================================================
# 4) 主入口
# ============================================================
CSV_FIELDS = [
    "run_id", "config_name", "row_type", "epoch", "n_epoch",
    "train_loss", "val_loss", "val_BER", "is_best", "elapsed_sec",
    "best_epoch", "best_val_loss", "best_val_BER",
    "final_train_loss", "final_val_loss", "final_val_BER", "total_time_sec",
    "N", "F", "F_prime", "T", "Nh1", "Nh2",
    "batch_size", "lr", "device", "model_type", "l2_weight",
]


def main():
    parser = argparse.ArgumentParser(description="GAT-OTFS Trainer (paper reproduction)")
    parser.add_argument(
        "--config", "-c", type=str,
        default=os.path.join(_python_root, "config", "gat_otfs_config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="使用合成数据 (不需要 MATLAB 生成的 .mat 文件)",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)

    # ---- 加载数据 ----
    raw = None
    N = int(config.get("data", {}).get("N_dim_half", 16))  # 默认 N=16

    if not args.synthetic and HAS_AFDM_DATA:
        try:
            paths = afdm_utils.resolve_dataset_paths(
                config, config_path, required_splits=("train", "val"),
            )
            meta = dataset_afdm.load_dataset_meta(paths["meta"])
            train_raw = dataset_afdm.load_afdm_split_mat(paths["train"], "train")
            val_raw = dataset_afdm.load_afdm_split_mat(paths["val"], "val")
            dataset_afdm.validate_split_against_meta(train_raw, "train", meta, paths["train"])
            dataset_afdm.validate_split_against_meta(val_raw, "val", meta, paths["val"])

            raw = {
                "x_daf_train": train_raw["x_daf_train"],
                "y_daf_train": train_raw["y_daf_train"],
                "H_eff_train": train_raw["H_eff_train"],
                "sigma2_train": train_raw["sigma2_train"],
                "n_train": train_raw["n_train"],
                "x_daf_val": val_raw["x_daf_val"],
                "y_daf_val": val_raw["y_daf_val"],
                "H_eff_val": val_raw["H_eff_val"],
                "sigma2_val": val_raw["sigma2_val"],
                "n_val": val_raw["n_val"],
                "N": train_raw["N"],
            }
            N = raw["N"]
            print(f"加载 AFDM 数据集: N={N}, n_train={raw['n_train']}, n_val={raw['n_val']}")
        except (ValueError, FileNotFoundError, KeyError) as e:
            print(f"Dataset/metadata validation failed: {e}")
            print("Falling back to synthetic mode due to data path or metadata issue.")
            raw = None

    if raw is None:
        print(f"使用合成数据模式: N={N}, n_dim={2*N}")

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 遍历配置 ----
    config_list = get_config_list(config)
    if not config_list:
        print("No valid configs found.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = os.path.join(_script_dir, f"experiment_log_gat_otfs_{run_id}.csv")

    write_header = True
    for i, cfg_entry in enumerate(config_list):
        cfg_name = cfg_entry.get("name", "unnamed")
        print(f"\n{'='*60}")
        print(f"配置 {i+1}/{len(config_list)}: {cfg_name}")
        print(f"{'='*60}")

        write_header = run_one_config(
            cfg_entry=cfg_entry,
            raw=raw,
            N=N,
            device=device,
            csv_path=csv_log_path,
            run_id=run_id,
            csv_fields=CSV_FIELDS,
            write_header=write_header,
        )

    print(f"\n训练全部完成. 日志: {csv_log_path}")


if __name__ == "__main__":
    main()
