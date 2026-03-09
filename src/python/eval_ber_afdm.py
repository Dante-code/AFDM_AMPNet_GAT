"""
Batch BER evaluation for AFDM AMP-GNN / AMP-GAT models.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import afdm_utils
import dataset_afdm
from amp_gat_detector import AMPGATDetector
from amp_gnn_detector import AMPGNNDetector, compute_ber
from graph_features import build_adjacency, build_edge_attr


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_list(config: dict) -> list:
    cfg_list = config.get("configs")
    if isinstance(cfg_list, list):
        return cfg_list
    if isinstance(config.get("model"), dict):
        return [{"name": "single", "model": config["model"]}]
    return []


def build_model(model_cfg: dict, n_dim: int, device: torch.device):
    model_type = model_cfg.get("model_type", "amp_gnn")
    n_iter = int(model_cfg.get("n_iter", 6))
    n_u = int(model_cfg.get("n_u", 12))
    n_h = int(model_cfg.get("n_h", 12))
    n_conv = int(model_cfg.get("n_conv", 2))
    n_mlp_hidden = int(model_cfg.get("n_mlp_hidden", 16))

    if model_type == "amp_gnn":
        model = AMPGNNDetector(
            n_dim=n_dim,
            n_iter=n_iter,
            n_u=n_u,
            n_h=n_h,
            n_conv=n_conv,
            n_mlp_hidden=n_mlp_hidden,
        ).to(device)
        return model, model_type

    if model_type == "amp_gat":
        edge_attr_mode = model_cfg.get("edge_attr_mode", "gram_triplet")
        edge_attr_dim = 3 if edge_attr_mode == "gram_triplet" else 1
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
            edge_attr_dim=edge_attr_dim,
            n_mlp_hidden=n_mlp_hidden,
        ).to(device)
        return model, model_type

    raise ValueError(f"Unsupported model_type: {model_type}")


def forward_model(model, model_type: str, model_cfg: dict, y: torch.Tensor, h: torch.Tensor, sigma2: torch.Tensor):
    adj = build_adjacency(h)
    if model_type == "amp_gat":
        edge_attr = None
        if bool(model_cfg.get("use_edge_attr", True)):
            edge_attr = build_edge_attr(h, mode=model_cfg.get("edge_attr_mode", "gram_triplet"))
        return model(y, h, sigma2, adj, edge_attr=edge_attr)
    return model(y, h, sigma2, adj)


def eval_one_config(
    cfg_entry: dict,
    raw: dict,
    n_dim: int,
    n: int,
    py_dir: str,
    device: torch.device,
) -> tuple[str, bool]:
    cfg_name = cfg_entry.get("name", "unnamed")
    model_cfg = cfg_entry.get("model", {})

    ckpt = os.path.join(py_dir, "..", f"{cfg_name}.pt")
    if not os.path.exists(ckpt):
        print(f"[{cfg_name}] 找不到对应.pt模型: {ckpt}")
        return "missing", False

    try:
        model, model_type = build_model(model_cfg, n_dim=n_dim, device=device)
    except Exception as e:
        print(f"[{cfg_name}] 配置错误，无法构建模型: {e}")
        return "failed", False

    payload = torch.load(ckpt, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[{cfg_name}] 模型权重与配置不匹配: {ckpt}")
        print(str(e))
        return "failed", False

    model.eval()
    n_test = raw["n_test"]
    snr_vec = raw["SNR_test_vec"]

    ber_by_snr: dict[int, list[float]] = {}
    with torch.no_grad():
        for i in range(n_test):
            x_vec = raw["x_daf_test"][i]
            y_vec = raw["y_daf_test"][i]
            h_eff = raw["H_eff_test"][i]
            sigma2 = raw["sigma2_test"][i]

            x, y, h, sigma2_r, _, _ = afdm_utils.prepare_sample(x_vec, y_vec, h_eff, sigma2)
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0).to(device)
            sigma2_t = torch.tensor([sigma2_r], dtype=torch.float32).to(device)

            x_hat = forward_model(model, model_type, model_cfg, y_t, h_t, sigma2_t)
            ber = compute_ber(x_t, x_hat, 1, n)
            snr = int(snr_vec[i])
            ber_by_snr.setdefault(snr, []).append(ber)

    print(f"\n[{cfg_name}] model_type={model_type} BER by SNR:")
    for snr in sorted(ber_by_snr.keys()):
        mean_ber = float(np.mean(ber_by_snr[snr]))
        print(f"  SNR={snr} dB: BER={mean_ber:.4e} (n={len(ber_by_snr[snr])})")
    return "ok", True


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate AFDM AMP-GNN / AMP-GAT BER")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to train config yaml",
    )
    args = parser.parse_args()

    py_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if args.config else os.path.join(py_dir, "config", "train_config.yaml")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    config_path = os.path.abspath(config_path)
    config = load_config(config_path)
    try:
        paths = afdm_utils.resolve_dataset_paths(config, config_path, required_splits=("test",))
        meta = dataset_afdm.load_dataset_meta(paths["meta"])
        raw = dataset_afdm.load_afdm_split_mat(paths["test"], "test")
        dataset_afdm.validate_split_against_meta(raw, "test", meta, paths["test"])
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Dataset config error: {e}")
        return

    config_list = get_config_list(config)
    if len(config_list) == 0:
        print(f"No valid configs found in {config_path}.")
        return

    n = raw["N"]
    n_dim = 2 * n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            n_dim=n_dim,
            n=n,
            py_dir=py_dir,
            device=device,
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
