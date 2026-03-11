"""
Batch BER evaluation for cmp_otfs_gat_paper checkpoints.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
sys.path.insert(0, _python_root)

import afdm_utils
import dataset_afdm
from gat_otfs_detector import GATOTFSDetector
from gat_otfs_loss import compute_ber_from_logits


class AFDMTestDataset(Dataset):
    def __init__(self, raw_data: dict):
        self.raw = raw_data
        self.n = raw_data["n_test"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_vec = self.raw["x_daf_test"][idx]
        y_vec = self.raw["y_daf_test"][idx]
        h_eff = self.raw["H_eff_test"][idx]
        sigma2 = self.raw["sigma2_test"][idx]
        snr = self.raw["SNR_test_vec"][idx]
        x, y, h, sigma2_r, _, _ = afdm_utils.prepare_sample(x_vec, y_vec, h_eff, sigma2)
        return x, y, h, sigma2_r, snr


def collate_fn(batch):
    x = np.stack([b[0] for b in batch])
    y = np.stack([b[1] for b in batch])
    h = np.stack([b[2] for b in batch])
    sigma2 = np.array([b[3] for b in batch])
    snr = np.array([b[4] for b in batch], dtype=np.int64)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(h, dtype=torch.float32),
        torch.tensor(sigma2, dtype=torch.float32),
        snr,
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_list(config: dict) -> list:
    if isinstance(config.get("configs"), list):
        return config["configs"]
    if isinstance(config.get("model"), dict):
        return [{"name": "single", "model": config["model"]}]
    return []


def build_detector(model_cfg: dict, device: torch.device) -> GATOTFSDetector:
    model = GATOTFSDetector(
        F=int(model_cfg.get("F", 8)),
        F_prime=int(model_cfg.get("F_prime", 16)),
        T=int(model_cfg.get("T", 10)),
        Nh1=int(model_cfg.get("Nh1", 64)),
        Nh2=int(model_cfg.get("Nh2", 32)),
        S=int(model_cfg.get("S", 2)),
        adj_eps=float(model_cfg.get("adj_eps", 1e-8)),
        add_self_loop=bool(model_cfg.get("add_self_loop", True)),
    ).to(device)
    return model


def load_test_data(config: dict, config_path: str) -> dict:
    paths = afdm_utils.resolve_dataset_paths(config, config_path, required_splits=("test",))
    meta = dataset_afdm.load_dataset_meta(paths["meta"])
    raw = dataset_afdm.load_afdm_split_mat(paths["test"], "test")
    dataset_afdm.validate_split_against_meta(raw, "test", meta, paths["test"])
    return raw


def eval_one_config(
    cfg_entry: dict,
    raw: dict,
    device: torch.device,
    checkpoint_dir: str,
    batch_size: int = 64,
) -> tuple[str, bool]:
    cfg_name = cfg_entry.get("name", "unnamed")
    model_cfg = cfg_entry.get("model", {})
    ckpt_path = os.path.join(checkpoint_dir, f"{cfg_name}.pt")

    if not os.path.exists(ckpt_path):
        print(f"[{cfg_name}] checkpoint not found: {ckpt_path}")
        return "missing", False

    model = build_detector(model_cfg, device=device)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[{cfg_name}] checkpoint/model mismatch: {ckpt_path}")
        print(str(e))
        return "failed", False

    model.eval()
    loader = DataLoader(AFDMTestDataset(raw), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    ber_by_snr: dict[int, list[float]] = {}

    with torch.no_grad():
        for x, y, h, sigma2, snr_np in loader:
            x = x.to(device)
            y = y.to(device)
            h = h.to(device)
            sigma2 = sigma2.to(device)

            logits = model(y, h, sigma2)
            # compute_ber_from_logits returns BER over the whole batch
            # to support SNR grouping, split by sample.
            for i in range(len(x)):
                ber_i = compute_ber_from_logits(logits[i : i + 1], x[i : i + 1])
                snr_i = int(snr_np[i])
                ber_by_snr.setdefault(snr_i, []).append(float(ber_i))

    print(f"\n[{cfg_name}] BER by SNR:")
    for snr in sorted(ber_by_snr):
        mean_ber = float(np.mean(ber_by_snr[snr]))
        print(f"  SNR={snr} dB: BER={mean_ber:.4e} (n={len(ber_by_snr[snr])})")
    return "ok", True


def main():
    parser = argparse.ArgumentParser(description="Evaluate BER for cmp_otfs_gat_paper checkpoints")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.path.join(_python_root, "config", "gat_otfs_config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(_script_dir, "checkpoints"),
        help="Directory containing <config_name>.pt checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    config_list = get_config_list(config)
    if len(config_list) == 0:
        print(f"No valid configs found in {config_path}")
        return

    try:
        raw = load_test_data(config, config_path)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Dataset config error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"n_test: {raw['n_test']}, N: {raw['N']}")

    total = len(config_list)
    ok_count = 0
    missing_count = 0
    failed_count = 0

    for i, cfg_entry in enumerate(config_list):
        cfg_name = cfg_entry.get("name", "unnamed")
        print(f"\n===== eval config {i+1}/{total}: {cfg_name} =====")
        status, ok = eval_one_config(
            cfg_entry=cfg_entry,
            raw=raw,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
        )
        if status == "ok" and ok:
            ok_count += 1
        elif status == "missing":
            missing_count += 1
        else:
            failed_count += 1

    print("\n===== Evaluation Summary =====")
    print(f"Total configs: {total}")
    print(f"Succeeded: {ok_count}")
    print(f"Missing checkpoint: {missing_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
