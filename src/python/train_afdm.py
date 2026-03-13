"""
Unified AFDM training entry:
- baseline AMP-GNN
- new AMP-GAT
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

_afdm_py = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _afdm_py)

import afdm_utils
import dataset_afdm
from amp_gnn_detector import AMPGNNDetector, compute_ber, compute_l2_loss
from amp_gat_detector import AMPGATDetector
from graph_features import build_adjacency, build_edge_attr


class AFDMDataset(Dataset):
    def __init__(self, raw_data, split: str):
        self.raw = raw_data
        self.split = split
        if split == "train":
            self.n = raw_data["n_train"]
        elif split == "val":
            self.n = raw_data["n_val"]
        else:
            self.n = raw_data["n_test"]

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


def collate_fn(batch):
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_list(config: dict) -> list:
    if "configs" in config:
        return config["configs"]
    return [{"name": "single", "model": config["model"], "train": config["train"]}]


def build_model(model_cfg: dict, n_dim: int):
    model_type = model_cfg.get("model_type", "amp_gnn")
    n_iter = model_cfg["n_iter"]
    n_u = model_cfg["n_u"]
    n_h = model_cfg["n_h"]
    n_conv = model_cfg["n_conv"]
    n_mlp_hidden = model_cfg["n_mlp_hidden"]

    if model_type == "amp_gnn":
        model = AMPGNNDetector(
            n_dim, n_iter=n_iter, n_u=n_u, n_h=n_h, n_conv=n_conv, n_mlp_hidden=n_mlp_hidden
        )
        return model

    if model_type == "amp_gat":
        model = AMPGATDetector(
            n_dim=n_dim,
            n_iter=n_iter,
            damp=float(model_cfg.get("damp", 0.7)),
            n_u=n_u,
            n_h=n_h,
            n_conv=n_conv,
            n_heads=int(model_cfg.get("n_heads", 2)),
            attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
            use_edge_attr=bool(model_cfg.get("use_edge_attr", True)),
            edge_attr_dim=3 if model_cfg.get("edge_attr_mode", "gram_triplet") == "gram_triplet" else 1,
            n_mlp_hidden=n_mlp_hidden,
        )
        return model

    raise ValueError(f"Unsupported model_type: {model_type}")


def forward_model(model, model_cfg: dict, y, H, sigma2):
    model_type = model_cfg.get("model_type", "amp_gnn")
    adj = build_adjacency(H)
    if model_type == "amp_gat":
        edge_attr = None
        if bool(model_cfg.get("use_edge_attr", True)):
            edge_attr = build_edge_attr(H, mode=model_cfg.get("edge_attr_mode", "gram_triplet"))
        return model(y, H, sigma2, adj, edge_attr=edge_attr)
    return model(y, H, sigma2, adj)


def run_one_config(
    cfg_entry: dict,
    raw: dict,
    n_dim: int,
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
    model_type = model_cfg.get("model_type", "amp_gnn")

    batch_size = train_cfg["batch_size"]
    val_batch_size = train_cfg["val_batch_size"]
    lr = train_cfg["lr"]
    n_epoch = train_cfg["n_epoch"]
    max_grad_norm = train_cfg["max_grad_norm"]
    scheduler_factor = train_cfg["scheduler_factor"]
    scheduler_patience = train_cfg["scheduler_patience"]

    train_loader = DataLoader(AFDMDataset(raw, "train"), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(AFDMDataset(raw, "val"), batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(model_cfg, n_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=scheduler_factor, patience=scheduler_patience)

    save_path = os.path.join(_afdm_py, "..", f"{cfg_name}.pt")
    best_val_loss = float("inf")
    best_val_ber = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(n_epoch):
        model.train()
        total_loss = 0.0
        for x, y, H, sigma2 in train_loader:
            x, y, H, sigma2 = x.to(device), y.to(device), H.to(device), sigma2.to(device)
            opt.zero_grad()
            x_hat = forward_model(model, model_cfg, y, H, sigma2)
            loss = compute_l2_loss(x, x_hat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            total_loss += loss.item()

        train_loss_avg = total_loss / max(1, len(train_loader))
        scheduler.step(train_loss_avg)

        model.eval()
        val_loss = 0.0
        val_ber = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for x, y, H, sigma2 in val_loader:
                x, y, H, sigma2 = x.to(device), y.to(device), H.to(device), sigma2.to(device)
                x_hat = forward_model(model, model_cfg, y, H, sigma2)
                val_loss += compute_l2_loss(x, x_hat).item()
                val_ber += compute_ber(x, x_hat, 1, N) * len(x)
                n_val_samples += len(x)

        val_loss_avg = val_loss / max(1, len(val_loader))
        val_ber_avg = val_ber / max(1, n_val_samples)

        is_best = 0
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_ber = val_ber_avg
            best_epoch = epoch + 1
            is_best = 1
            torch.save(model.state_dict(), save_path)

        elapsed_sec = time.time() - t0
        row = {k: "" for k in csv_fields}
        row.update(
            {
                "run_id": run_id,
                "config_name": cfg_name,
                "row_type": "epoch",
                "epoch": epoch + 1,
                "n_epoch": n_epoch,
                "train_loss": f"{train_loss_avg:.6f}",
                "val_loss": f"{val_loss_avg:.6f}",
                "val_BER": f"{val_ber_avg:.6e}",
                "is_best": is_best,
                "elapsed_sec": f"{elapsed_sec:.1f}",
                "model_type": model_type,
                "damp": model_cfg.get("damp", ""),
                "n_heads": model_cfg.get("n_heads", ""),
                "use_edge_attr": model_cfg.get("use_edge_attr", ""),
                "edge_attr_mode": model_cfg.get("edge_attr_mode", ""),
            }
        )
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        print(
            f"[{cfg_name}] epoch {epoch+1}/{n_epoch} "
            f"loss={train_loss_avg:.4f} val_loss={val_loss_avg:.4f} val_BER={val_ber_avg:.4e}"
        )

    total_time_sec = time.time() - t0
    summary = {k: "" for k in csv_fields}
    summary.update(
        {
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
            "total_time_sec": f"{total_time_sec:.1f}",
            "N": N,
            "P": raw["P"],
            "lr": lr,
            "n_iter": model_cfg["n_iter"],
            "n_u": model_cfg["n_u"],
            "n_h": model_cfg["n_h"],
            "n_conv": model_cfg["n_conv"],
            "n_mlp_hidden": model_cfg["n_mlp_hidden"],
            "batch_size": batch_size,
            "device": str(device),
            "model_type": model_type,
            "damp": model_cfg.get("damp", ""),
            "n_heads": model_cfg.get("n_heads", ""),
            "use_edge_attr": model_cfg.get("use_edge_attr", ""),
            "edge_attr_mode": model_cfg.get("edge_attr_mode", ""),
        }
    )
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writerow(summary)


def main():
    parser = argparse.ArgumentParser(description="AFDM unified trainer for AMP-GNN / AMP-GAT")
    parser.add_argument(
        "--config", "-c", type=str, default=os.path.join(_afdm_py, "config", "train_config.yaml")
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)

    try:
        paths = afdm_utils.resolve_dataset_paths(config, config_path, required_splits=("train", "val"))
        meta = dataset_afdm.load_dataset_meta(paths["meta"])
        train_raw = dataset_afdm.load_afdm_split_mat(paths["train"], "train")
        val_raw = dataset_afdm.load_afdm_split_mat(paths["val"], "val")
        dataset_afdm.validate_split_against_meta(train_raw, "train", meta, paths["train"])
        dataset_afdm.validate_split_against_meta(val_raw, "val", meta, paths["val"])
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Dataset config error: {e}")
        return

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
        "P": train_raw.get("P", meta.get("common", {}).get("P")),
        "QAM_order": train_raw.get("QAM_order", meta.get("common", {}).get("QAM_order")),
    }
    N = raw["N"]
    n_dim = 2 * N
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_list = get_config_list(config)

    seen_names: set[str] = set()
    dup_names: set[str] = set()
    for cfg in config_list:
        name = cfg.get("name", "unnamed")
        if name in seen_names:
            dup_names.add(name)
        seen_names.add(name)
    if dup_names:
        print(f"Warning: duplicate config name(s) detected: {sorted(dup_names)}. Later runs will overwrite same .pt file.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = os.path.join(_afdm_py, "..", f"experiment_log_afdm_{run_id}.csv")
    csv_fields = [
        "run_id",
        "config_name",
        "row_type",
        "epoch",
        "n_epoch",
        "train_loss",
        "val_loss",
        "val_BER",
        "is_best",
        "elapsed_sec",
        "best_epoch",
        "best_val_loss",
        "best_val_BER",
        "final_train_loss",
        "final_val_loss",
        "final_val_BER",
        "total_time_sec",
        "N",
        "P",
        "lr",
        "n_iter",
        "n_u",
        "n_h",
        "n_conv",
        "n_mlp_hidden",
        "batch_size",
        "device",
        "model_type",
        "damp",
        "n_heads",
        "use_edge_attr",
        "edge_attr_mode",
    ]

    write_header = True
    for i, cfg_entry in enumerate(config_list):
        print(f"\n===== config {i+1}/{len(config_list)}: {cfg_entry.get('name','unnamed')} =====")
        run_one_config(
            cfg_entry=cfg_entry,
            raw=raw,
            n_dim=n_dim,
            N=N,
            device=device,
            csv_path=csv_log_path,
            run_id=run_id,
            csv_fields=csv_fields,
            write_header=write_header,
        )
        write_header = False

    print(f"All done. Log saved to {csv_log_path}")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
