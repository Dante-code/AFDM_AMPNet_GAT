"""
AFDM 实值转换与索引集合构造
AFDM DAF 域为 1D 向量 (N,)，x = [Re(x̄)^T, Im(x̄)^T]^T, H 实值扩展
"""
import os
import numpy as np


def complex_to_real_vector(z: np.ndarray) -> np.ndarray:
    """x = [Re(z)^T, Im(z)^T]^T"""
    return np.concatenate([z.real, z.imag], axis=0)


def complex_to_real_matrix(H: np.ndarray) -> np.ndarray:
    """H = [[Re(H̄), -Im(H̄)]; [Im(H̄), Re(H̄)]]"""
    Hr = H.real
    Hi = H.imag
    top = np.hstack([Hr, -Hi])
    bot = np.hstack([Hi, Hr])
    return np.vstack([top, bot])


def real_to_complex_vector(x: np.ndarray) -> np.ndarray:
    """x = [Re^T, Im^T]^T -> z = Re + j*Im"""
    n = len(x) // 2
    return x[:n] + 1j * x[n:]


def build_index_sets(H: np.ndarray, eps: float = 1e-12):
    """
    从实值 H 构造 I(j) 和 L(i)
    I(j): 第 j 行非零列索引
    L(i): 第 i 列非零行索引
    返回: I_list, L_list (list of arrays)
    """
    n = H.shape[0]
    I_list = []
    L_list = []
    for j in range(n):
        I_list.append(np.where(np.abs(H[j, :]) > eps)[0])
    for i in range(n):
        L_list.append(np.where(np.abs(H[:, i]) > eps)[0])
    return I_list, L_list


def build_idi_mask_from_loc_main(
    H_real: np.ndarray,
    loc_main: np.ndarray,
    N: int,
    kv: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    根据 loc_main 和 kv 构建实值域 IDI 掩码。
    返回 mask_idi: (2N, 2N), True 表示扩展项（IDI）位置。
    """
    n = 2 * N
    if kv <= 0:
        return np.zeros((n, n), dtype=bool)

    loc_main = np.asarray(loc_main).reshape(-1).astype(np.int64)
    p_idx = np.arange(N, dtype=np.int64).reshape(-1, 1)  # (N,1)
    q_main = (p_idx + loc_main.reshape(1, -1)) % N  # (N,P)

    main_mask_complex = np.zeros((N, N), dtype=bool)
    row_idx = np.repeat(np.arange(N, dtype=np.int64), q_main.shape[1])
    col_idx = q_main.reshape(-1)
    main_mask_complex[row_idx, col_idx] = True

    main_mask_real = np.zeros((n, n), dtype=bool)
    main_mask_real[:N, :N] = main_mask_complex
    main_mask_real[:N, N:] = main_mask_complex
    main_mask_real[N:, :N] = main_mask_complex
    main_mask_real[N:, N:] = main_mask_complex

    nonzero_mask = np.abs(H_real) > eps
    mask_idi = nonzero_mask & (~main_mask_real)
    return mask_idi


def prepare_sample(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    H_eff: np.ndarray,
    sigma2: float,
    loc_main: np.ndarray | None = None,
    N: int | None = None,
    kv: int = 0,
    return_mask: bool = False,
):
    """
    将 (x_vec, y_vec, H_eff) 转为论文实值形式
    x_vec, y_vec: (N,) DAF 域 1D 向量
    返回:
      - 默认: x, y, H, sigma2, I_list, L_list
      - return_mask=True: 额外返回 mask_idi
    """
    x_vec = np.asarray(x_vec).flatten()
    y_vec = np.asarray(y_vec).flatten()

    x = complex_to_real_vector(x_vec)
    y = complex_to_real_vector(y_vec)
    H = complex_to_real_matrix(H_eff)

    sigma2_real = sigma2 / 2.0
    I_list, L_list = build_index_sets(H)
    if not return_mask:
        return x, y, H, sigma2_real, I_list, L_list

    mask_idi = None
    n_complex = len(x_vec) if N is None else int(N)
    if loc_main is not None and int(kv) > 0:
        mask_idi = build_idi_mask_from_loc_main(H, np.asarray(loc_main), n_complex, int(kv))
    return x, y, H, sigma2_real, I_list, L_list, mask_idi


def _resolve_required_path(data_cfg: dict, config_dir: str, key: str) -> str:
    value = data_cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required config field: data.{key}")
    value = value.strip()
    resolved = value if os.path.isabs(value) else os.path.abspath(os.path.join(config_dir, value))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Path not found for data.{key}: {resolved}")
    return resolved


def resolve_dataset_paths(config: dict, config_path: str, required_splits: tuple[str, ...]) -> dict:
    """
    Resolve split dataset paths and metadata path from config.
    Required config fields:
      data.train_dataset_path, data.val_dataset_path, data.test_dataset_path, data.dataset_meta_path
    Relative paths are resolved against the config file directory.
    """
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Missing required config section: data")

    if "dataset_path" in data_cfg:
        raise ValueError(
            "Deprecated config field detected: data.dataset_path. "
            "Use data.train_dataset_path / data.val_dataset_path / data.test_dataset_path / data.dataset_meta_path."
        )

    config_dir = os.path.dirname(os.path.abspath(config_path))
    out = {
        "meta": _resolve_required_path(data_cfg, config_dir, "dataset_meta_path"),
    }
    split_to_key = {
        "train": "train_dataset_path",
        "val": "val_dataset_path",
        "test": "test_dataset_path",
    }
    for split in required_splits:
        if split not in split_to_key:
            raise ValueError(f"Unsupported required split: {split}")
        out[split] = _resolve_required_path(data_cfg, config_dir, split_to_key[split])
    return out
