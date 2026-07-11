"""Dataset utilities for MIONet-FSS.

Provides the PyTorch Dataset class for TE+TM dual-polarization FSS data,
dataset splitting by geometry ID, and frequency tensor construction.
"""
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DATASET_PATH as _DEFAULT_DATASET_PATH


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_mode_id(mode_names, target_mode_name):
    """Return the integer mode id for *target_mode_name* ("TE" or "TM")."""
    for idx, name in enumerate(mode_names):
        if str(name).upper() == str(target_mode_name).upper():
            return idx
    if str(target_mode_name).upper() == "TE":
        return 0
    if str(target_mode_name).upper() == "TM":
        return 1
    raise ValueError(f"Unknown mode name: {target_mode_name}")


def _build_pair_records(data, theta_min_deg=None, theta_max_deg=None):
    """Build a list of paired TE+TM records from the raw data dict.

    Each record is a dict with keys:
        matrix_idx, te_idx, tm_idx, theta_deg, geometry_id
    """
    matrices = data["matrices"].float()
    theta_deg = data["theta_deg"].float()
    mode_id = data["mode_id"].long()
    mode_names = data["meta"].get("mode_names", ["TE", "TM"])

    te_id = _get_mode_id(mode_names, "TE")
    tm_id = _get_mode_id(mode_names, "TM")
    pair_map = {}

    for idx in range(matrices.shape[0]):
        theta_val = float(theta_deg[idx].item())

        if theta_min_deg is not None and theta_val < float(theta_min_deg):
            continue
        if theta_max_deg is not None and theta_val > float(theta_max_deg):
            continue

        mat_np = matrices[idx].cpu().numpy().astype(np.float32, copy=False)
        key = (mat_np.tobytes(), round(theta_val, 8))
        geometry_id = hashlib.sha1(mat_np.tobytes()).hexdigest()

        if key not in pair_map:
            pair_map[key] = {
                "matrix_idx": idx,
                "theta_deg": theta_val,
                "te_idx": None,
                "tm_idx": None,
                "geometry_id": geometry_id,
            }

        mode_val = int(mode_id[idx].item())
        if mode_val == te_id:
            pair_map[key]["te_idx"] = idx
        elif mode_val == tm_id:
            pair_map[key]["tm_idx"] = idx

    pair_records = []
    for rec in pair_map.values():
        if rec["te_idx"] is not None and rec["tm_idx"] is not None:
            pair_records.append(rec)
    return pair_records


# ---------------------------------------------------------------------------
# Lightweight container for the loaded dataset
# ---------------------------------------------------------------------------

class _DataObj:
    """Simple namespace holding dataset tensors and metadata."""
    pass


# ---------------------------------------------------------------------------
# Frequency tensor builder
# ---------------------------------------------------------------------------

def build_freq_tensor(norm_freqs, device=None):
    """Build a frequency tensor from normalized frequency values.

    Parameters
    ----------
    norm_freqs : np.ndarray
        Normalized frequency grid, shape (N_FREQ,).
    device : torch.device or None
        Target device.  If None, CPU is used.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape (N_FREQ,).
    """
    return torch.tensor(norm_freqs, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ForwardDualModeThetaMagDataset(Dataset):
    """PyTorch Dataset for dual-polarization (TE+TM) FSS magnitude data.

    Each sample pairs a metalisation geometry with the four S-parameter
    magnitude curves: TE S11, TE S21, TM S11, TM S21.

    Parameters
    ----------
    data_obj : _DataObj
        Container with ``.matrices``, ``.targets`` and ``.norm_freqs``.
    pair_records : list[dict]
        List of paired TE+TM records, each with keys ``matrix_idx``,
        ``te_idx``, ``tm_idx``, ``theta_deg``, ``geometry_id``.
    """

    def __init__(self, data_obj, pair_records):
        self.data_obj = data_obj
        self.pair_records = pair_records

    def __len__(self):
        return len(self.pair_records)

    def __getitem__(self, idx):
        rec = self.pair_records[idx]

        # Geometry matrix
        geometry_tensor = self.data_obj.matrices[rec["matrix_idx"]].float()

        # TE targets --- real+imag -> magnitude
        te_raw = self.data_obj.targets[rec["te_idx"]]
        te_s11 = torch.sqrt(te_raw[0] ** 2 + te_raw[1] ** 2 + 1e-8).clamp(0.0, 1.0)
        te_s21 = torch.sqrt(te_raw[2] ** 2 + te_raw[3] ** 2 + 1e-8).clamp(0.0, 1.0)

        # TM targets
        tm_raw = self.data_obj.targets[rec["tm_idx"]]
        tm_s11 = torch.sqrt(tm_raw[0] ** 2 + tm_raw[1] ** 2 + 1e-8).clamp(0.0, 1.0)
        tm_s21 = torch.sqrt(tm_raw[2] ** 2 + tm_raw[3] ** 2 + 1e-8).clamp(0.0, 1.0)

        return geometry_tensor, (te_s11, te_s21, tm_s11, tm_s21)


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def load_pair_splits(
    pt_path=None,
    train_ratio=0.8,
    seed=42,
    theta_min_deg=0,
    theta_max_deg=60,
    split_by_geometry=True,
):
    """Load the .pt dataset, filter by theta range, and split by geometry ID.

    Parameters
    ----------
    pt_path : str or None
        Path to the ``.pt`` file.  Defaults to ``DATASET_PATH`` from config.
    train_ratio : float
        Fraction of geometries used for training.
    seed : int
        Random seed for the split.
    theta_min_deg : float
        Minimum incident angle in degrees (inclusive).
    theta_max_deg : float
        Maximum incident angle in degrees (inclusive).
    split_by_geometry : bool
        If True, split at the geometry level so the same geometry never
        appears in both train and val sets.

    Returns
    -------
    data_obj : _DataObj
        Container with ``.matrices``, ``.targets``, ``.norm_freqs``.
    train_pairs : list[dict]
        Training pair records.
    val_pairs : list[dict]
        Validation pair records.
    """
    if pt_path is None:
        pt_path = _DEFAULT_DATASET_PATH

    if not os.path.isabs(pt_path):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), pt_path)
        if os.path.exists(candidate):
            pt_path = candidate

    print(f"Loading paired TE/TM dataset from: {pt_path}")
    raw = torch.load(pt_path, weights_only=False)

    # Build data object --------------------------------------------------
    data_obj = _DataObj()
    data_obj.matrices = raw["matrices"].float()
    data_obj.targets = raw["targets"].float()
    freqs = raw["freqs"].float()

    # Normalised frequency grid -----------------------------------------
    freq_min = float(raw["meta"].get("freq_min", freqs.min().item()))
    freq_max = float(raw["meta"].get("freq_max", freqs.max().item()))
    standard_freqs = freqs.cpu().numpy().astype(np.float32)
    data_obj.norm_freqs = (
        2.0 * (standard_freqs - freq_min) / (freq_max - freq_min) - 1.0
    ).astype(np.float32)

    # Build pair records -------------------------------------------------
    pair_records = _build_pair_records(
        raw,
        theta_min_deg=theta_min_deg,
        theta_max_deg=theta_max_deg,
    )

    if not pair_records:
        desc = f"theta in [{theta_min_deg}, {theta_max_deg}]"
        raise RuntimeError(f"No paired TE/TM samples found for {desc}.")

    # Split --------------------------------------------------------------
    generator = torch.Generator().manual_seed(seed)

    if split_by_geometry:
        geometry_ids = sorted({rec["geometry_id"] for rec in pair_records})
        perm = torch.randperm(len(geometry_ids), generator=generator).tolist()
        geometry_ids = [geometry_ids[i] for i in perm]
        train_size = int(train_ratio * len(geometry_ids))
        train_geometry = set(geometry_ids[:train_size])
        train_pairs = [rec for rec in pair_records if rec["geometry_id"] in train_geometry]
        val_pairs = [rec for rec in pair_records if rec["geometry_id"] not in train_geometry]
    else:
        perm = torch.randperm(len(pair_records), generator=generator).tolist()
        pair_records = [pair_records[i] for i in perm]
        train_size = int(train_ratio * len(pair_records))
        train_pairs = pair_records[:train_size]
        val_pairs = pair_records[train_size:]

    print(
        f"Paired samples loaded: total={len(pair_records)}, "
        f"train={len(train_pairs)}, val={len(val_pairs)}, "
        f"split_by_geometry={split_by_geometry}"
    )
    return data_obj, train_pairs, val_pairs
