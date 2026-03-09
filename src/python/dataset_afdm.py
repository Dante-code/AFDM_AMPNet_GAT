"""
AFDM dataset loaders for split-based files and metadata yaml.
"""
from __future__ import annotations

import os
import numpy as np
import h5py
import yaml
from scipy.io import loadmat

import afdm_utils


def _load_scalar_from_h5(f, name: str):
    arr = np.array(f[name])
    return arr.reshape(-1)[0]


def _maybe_transpose_2d(arr: np.ndarray, n: int, N: int) -> np.ndarray:
    if arr.shape == (n, N):
        return arr
    if arr.shape == (N, n):
        return np.transpose(arr)
    return arr


def _to_complex(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    return arr


def _to_complex64_large(arr: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    if arr.dtype.fields is None or "real" not in arr.dtype.fields or "imag" not in arr.dtype.fields:
        return arr.astype(np.complex64) if arr.dtype != np.complex64 else arr
    n = arr.shape[0]
    out = np.empty(arr.shape, dtype=np.complex64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = arr["real"][start:end].astype(np.float32) + 1j * arr["imag"][start:end].astype(np.float32)
    return out


def _split_keys(split: str) -> dict[str, str]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return {
        "x": f"x_daf_{split}_arr",
        "y": f"y_daf_{split}_arr",
        "h": f"H_eff_{split}_arr",
        "sigma2": f"sigma2_{split}",
        "n": f"n_{split}",
    }


def _extract_common_from_mat_dict(data: dict) -> dict:
    out = {}
    for key in ("N", "P", "QAM_order", "l_max", "k_max", "SNR_dB", "c1", "c2"):
        if key in data:
            out[key] = int(data[key].flatten()[0]) if key in {"N", "P", "QAM_order", "l_max", "k_max"} else float(
                data[key].flatten()[0]
            )
    return out


def _extract_common_from_h5(f) -> dict:
    out = {}
    for key in ("N", "P", "QAM_order", "l_max", "k_max", "SNR_dB", "c1", "c2"):
        if key in f:
            val = _load_scalar_from_h5(f, key)
            out[key] = int(val) if key in {"N", "P", "QAM_order", "l_max", "k_max"} else float(val)
    return out


def load_afdm_split_mat(path: str, split: str) -> dict:
    """
    Load one split dataset file (train/val/test).
    Returns normalized keys:
      - x_daf_{split}, y_daf_{split}, H_eff_{split}, sigma2_{split}, n_{split}
      - optional: SNR_test_vec (for split=test)
      - common: N, P, QAM_order, l_max, k_max, c1, c2 (if present)
    """
    keys = _split_keys(split)
    try:
        data = loadmat(path)
        n = int(data[keys["n"]].flatten()[0])
        N = int(data["N"].flatten()[0])
        out = {
            f"x_daf_{split}": data[keys["x"]],
            f"y_daf_{split}": data[keys["y"]],
            f"H_eff_{split}": data[keys["h"]],
            f"sigma2_{split}": data[keys["sigma2"]].flatten(),
            keys["n"]: n,
            "N": N,
        }
        out.update(_extract_common_from_mat_dict(data))
        if split == "test":
            if "SNR_test_vec" not in data:
                raise ValueError(f"{path} missing required key for test split: SNR_test_vec")
            out["SNR_test_vec"] = data["SNR_test_vec"].flatten()
        return out
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as f:
        n = int(_load_scalar_from_h5(f, keys["n"]))
        N = int(_load_scalar_from_h5(f, "N"))

        x = _maybe_transpose_2d(np.array(f[keys["x"]]), n, N)
        y = _maybe_transpose_2d(np.array(f[keys["y"]]), n, N)
        h = np.array(f[keys["h"]])
        if h.shape == (N, N, n):
            h = np.transpose(h, (2, 0, 1))
            h = np.transpose(h, (0, 2, 1))

        x = _to_complex(x)
        y = _to_complex(y)
        h = _to_complex64_large(h)
        sigma2 = np.array(f[keys["sigma2"]]).reshape(-1)

        out = {
            f"x_daf_{split}": x,
            f"y_daf_{split}": y,
            f"H_eff_{split}": h,
            f"sigma2_{split}": sigma2,
            keys["n"]: n,
            "N": N,
        }
        out.update(_extract_common_from_h5(f))
        if split == "test":
            if "SNR_test_vec" not in f:
                raise ValueError(f"{path} missing required key for test split: SNR_test_vec")
            out["SNR_test_vec"] = np.array(f["SNR_test_vec"]).reshape(-1)
        return out


def load_dataset_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    if not isinstance(meta, dict) or "dataset" not in meta:
        raise ValueError(f"Invalid metadata yaml: {meta_path} (missing top-level key: dataset)")
    return meta["dataset"]


def validate_split_against_meta(split_data: dict, split: str, meta: dict, data_path: str) -> None:
    common = meta.get("common")
    splits = meta.get("splits")
    if not isinstance(common, dict):
        raise ValueError("Metadata missing key: dataset.common")
    if not isinstance(splits, dict):
        raise ValueError("Metadata missing key: dataset.splits")
    if split not in splits:
        raise ValueError(f"Metadata missing split section: dataset.splits.{split}")
    expected_file = splits[split].get("file")
    if not isinstance(expected_file, str) or not expected_file.strip():
        raise ValueError(f"Metadata missing key: dataset.splits.{split}.file")
    actual_file = os.path.basename(data_path)
    if actual_file != expected_file:
        raise ValueError(
            f"File mismatch for split={split}: config path points to {actual_file}, "
            f"but metadata expects {expected_file} (file: {data_path})"
        )

    for key in ("N", "P", "QAM_order", "l_max", "k_max"):
        if key not in common:
            raise ValueError(f"Metadata missing key: dataset.common.{key}")
        if key not in split_data:
            raise ValueError(f"Split data missing key in {data_path}: {key}")
        if int(split_data[key]) != int(common[key]):
            raise ValueError(
                f"Parameter mismatch for {key}: split file={split_data[key]} vs metadata={common[key]} (file: {data_path})"
            )
    if split in {"train", "val"}:
        if "snr_train_db" not in common:
            raise ValueError("Metadata missing key: dataset.common.snr_train_db")
        if "SNR_dB" not in split_data:
            raise ValueError(f"Split data missing key in {data_path}: SNR_dB")
        if int(round(float(split_data["SNR_dB"]))) != int(round(float(common["snr_train_db"]))):
            raise ValueError(
                "Parameter mismatch for snr_train_db: "
                f"split file={split_data['SNR_dB']} vs metadata={common['snr_train_db']} (file: {data_path})"
            )

    split_count = split_data.get(f"n_{split}")
    meta_count = splits[split].get("count")
    if split_count is None:
        raise ValueError(f"Split data missing count key: n_{split} (file: {data_path})")
    if meta_count is None:
        raise ValueError(f"Metadata missing key: dataset.splits.{split}.count")
    if int(split_count) != int(meta_count):
        raise ValueError(
            f"Count mismatch for split={split}: split file={split_count} vs metadata={meta_count} (file: {data_path})"
        )

    if split == "test":
        snr_vec = split_data.get("SNR_test_vec")
        if snr_vec is None:
            raise ValueError(f"Split data missing key for test split: SNR_test_vec (file: {data_path})")
        meta_snr = splits["test"].get("snr_test_db")
        if meta_snr is None:
            raise ValueError("Metadata missing key: dataset.splits.test.snr_test_db")
        snr_unique = sorted({int(v) for v in np.asarray(snr_vec).reshape(-1)})
        meta_unique = sorted({int(v) for v in meta_snr})
        if snr_unique != meta_unique:
            raise ValueError(
                f"SNR mismatch for test split: split file={snr_unique} vs metadata={meta_unique} (file: {data_path})"
            )


def prepare_batch(raw_data, split: str, indices, utils_module=afdm_utils):
    x_vec = raw_data[f"x_daf_{split}"][indices]
    y_vec = raw_data[f"y_daf_{split}"][indices]
    H_eff = raw_data[f"H_eff_{split}"][indices]
    sigma2 = raw_data[f"sigma2_{split}"][indices]
    bsz = len(indices)
    n = raw_data["N"] * 2
    x_batch = np.zeros((bsz, n), dtype=np.float64)
    y_batch = np.zeros((bsz, n), dtype=np.float64)
    H_batch = np.zeros((bsz, n, n), dtype=np.float64)
    sigma2_batch = np.zeros(bsz, dtype=np.float64)
    for i in range(bsz):
        x, y, H, sigma2_r, _, _ = utils_module.prepare_sample(x_vec[i], y_vec[i], H_eff[i], sigma2[i])
        x_batch[i] = x
        y_batch[i] = y
        H_batch[i] = H
        sigma2_batch[i] = sigma2_r
    return x_batch, y_batch, H_batch, sigma2_batch
