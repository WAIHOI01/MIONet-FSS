"""Dataset oracle search for best design candidates.

Brute-force traversal over the validation set, using ground-truth
S-parameters rather than the world model.
"""

import os
from collections import defaultdict

import numpy as np
import torch

from common.config import DEVICE, N_FREQ, STANDARD_FREQS, DATASET_PATH
from common.dataset import (
    ForwardDualModeThetaMagDataset,
    load_pair_splits,
    build_freq_tensor,
)
from inverse.reward import calculate_sparam_reward_v11, aggregate_stability_reward
from inverse.task import build_active_design_task


# ===================================================================
#  Internal helpers
# ===================================================================

def _build_mag_targets(targets, idx):
    """Reconstruct magnitude targets for a given sample index.

    Returns ``(s11, s21)`` as (N_FREQ,) float32 numpy arrays clipped to
    [0, 1].
    """
    raw = targets[idx]
    s11 = (
        torch.sqrt(raw[0] ** 2 + raw[1] ** 2 + 1e-8)
        .clamp(0.0, 1.0)
        .numpy()
        .astype(np.float32)
        .flatten()
    )
    s21 = (
        torch.sqrt(raw[2] ** 2 + raw[3] ** 2 + 1e-8)
        .clamp(0.0, 1.0)
        .numpy()
        .astype(np.float32)
        .flatten()
    )
    return s11, s21


def _score_single(s11, s21, freqs_ghz, spec):
    """Score one spec and return the reward."""
    reward, _, _ = calculate_sparam_reward_v11(
        np.abs(s11), np.abs(s21), freqs_ghz, spec,
    )
    return float(reward)


def _build_freqs_ghz(data_obj):
    """Build a GHz frequency grid matching the dataset.

    Falls back to ``STANDARD_FREQS`` if the dataset frequency count matches.
    """
    n_freq = data_obj.targets.shape[-1]
    if n_freq == N_FREQ:
        return STANDARD_FREQS.copy()
    return np.linspace(0.01, 20.0, n_freq, dtype=np.float32)


# ===================================================================
#  Polar-selective (mode) search
# ===================================================================

def search_dataset_mode(val_ds, freqs_ghz, fc, bw):
    """Search the validation set for the best polar-selective (mode) candidate.

    Only considers records at theta = 0 deg.

    Parameters
    ----------
    val_ds : ForwardDualModeThetaMagDataset
    freqs_ghz : ndarray
    fc : float
    bw : float

    Returns
    -------
    best_pattern : ndarray or None  (10, 10)
    best_pred_items : list of dict
    best_reward : float
    best_idx : int
    all_rewards : list of float
    """
    specs, _ = build_active_design_task(fc, bw, "mode", "transmit")
    targets = val_ds.data_obj.targets
    matrices = val_ds.data_obj.matrices

    best_reward = -float("inf")
    best_idx = -1
    best_pattern = None
    best_pred_items = None
    all_rewards = []

    for idx, rec in enumerate(val_ds.pair_records):
        theta = float(rec["theta_deg"])
        if abs(theta) > 1e-6:
            continue

        mat = matrices[rec["matrix_idx"]].float()
        te_s11, te_s21 = _build_mag_targets(targets, rec["te_idx"])
        tm_s11, tm_s21 = _build_mag_targets(targets, rec["tm_idx"])

        pred_items = []
        rewards = []
        for spec in specs:
            pol = int(spec["pol"])
            if pol == 0:
                s11, s21 = te_s11, te_s21
            else:
                s11, s21 = tm_s11, tm_s21
            reward = _score_single(s11, s21, freqs_ghz, spec)
            rewards.append(reward)
            pred_items.append({
                "pol": pol,
                "theta": 0.0,
                "type": spec["type"],
                "label": spec["label"],
                "s11": s11.astype(np.float32),
                "s21": s21.astype(np.float32),
            })

        total = aggregate_stability_reward(rewards, specs, stability_weight=0.25)
        all_rewards.append(total)
        if total > best_reward:
            best_reward = total
            best_idx = idx
            best_pattern = mat.squeeze().numpy()
            best_pred_items = pred_items

    return best_pattern, best_pred_items, best_reward, best_idx, all_rewards


# ===================================================================
#  Geometry-grouped search (angle / joint)
# ===================================================================

def _evaluate_geometry(val_ds, indexed_records, freqs_ghz, fc, bw, task_kind):
    """Evaluate one geometry across all its theta variants.

    Parameters
    ----------
    val_ds : ForwardDualModeThetaMagDataset
    indexed_records : list of (idx, rec)
    freqs_ghz : ndarray
    fc, bw : float
    task_kind : str

    Returns
    -------
    pattern, pred_items, total_reward, rewards
    """
    specs, _ = build_active_design_task(fc, bw, task_kind, "transmit")
    targets = val_ds.data_obj.targets
    matrices = val_ds.data_obj.matrices

    # Build theta-indexed lookup
    by_theta = {}
    for idx, rec in indexed_records:
        theta = round(float(rec["theta_deg"]), 8)
        by_theta[theta] = (idx, rec)

    pred_items = []
    rewards = []
    for spec in specs:
        theta = round(float(spec["theta"]), 8)
        match = by_theta.get(theta)
        if match is None:
            continue
        idx, rec = match
        te_s11, te_s21 = _build_mag_targets(targets, rec["te_idx"])
        tm_s11, tm_s21 = _build_mag_targets(targets, rec["tm_idx"])

        pol = int(spec["pol"])
        if pol == 0:
            s11, s21 = te_s11, te_s21
        else:
            s11, s21 = tm_s11, tm_s21
        reward = _score_single(s11, s21, freqs_ghz, spec)
        rewards.append(reward)
        pred_items.append({
            "pol": pol,
            "theta": float(theta),
            "type": spec["type"],
            "label": spec["label"],
            "s11": s11.astype(np.float32),
            "s21": s21.astype(np.float32),
        })

    total_reward = aggregate_stability_reward(
        rewards, specs, stability_weight=0.25,
    )
    # Take the pattern from the first record of this geometry
    first_rec = indexed_records[0][1]
    pattern = (
        matrices[first_rec["matrix_idx"]].float().squeeze().numpy()
    )
    return pattern, pred_items, total_reward, rewards


def search_dataset_by_geometry(val_ds, freqs_ghz, fc, bw, task_kind):
    """Search the validation set grouped by geometry ID.

    For angle/joint tasks where multiple theta values must be satisfied
    by the same physical pattern.

    Parameters
    ----------
    val_ds : ForwardDualModeThetaMagDataset
    freqs_ghz : ndarray
    fc, bw : float
    task_kind : str  ("angle" or "joint")

    Returns
    -------
    best_pattern : ndarray or None
    best_pred_items : list of dict
    best_reward : float
    best_gid : str
    all_rewards : list of float
    """
    # Group records by geometry_id
    groups = defaultdict(list)
    for idx, rec in enumerate(val_ds.pair_records):
        gid = rec["geometry_id"]
        theta = float(rec["theta_deg"])
        groups[gid].append((idx, theta, rec))

    best_reward = -float("inf")
    best_pattern = None
    best_pred_items = None
    best_gid = None
    all_rewards = []

    for gid, records in groups.items():
        indexed = [(item[0], item[2]) for item in records]
        try:
            pattern, pred_items, reward, _ = _evaluate_geometry(
                val_ds, indexed, freqs_ghz, fc, bw, task_kind,
            )
        except Exception:
            continue
        all_rewards.append(reward)
        if reward > best_reward:
            best_reward = reward
            best_pattern = pattern
            best_pred_items = pred_items
            best_gid = gid

    return best_pattern, best_pred_items, best_reward, best_gid, all_rewards


# ===================================================================
#  Unified entry point
# ===================================================================

def search_dataset(fc, bw, task_kind, dataset_path=None):
    """Unified entry point for dataset oracle search.

    Loads the validation split, then dispatches to the appropriate
    search method based on *task_kind*.

    Parameters
    ----------
    fc : float
        Centre frequency in GHz.
    bw : float
        Bandwidth in GHz.
    task_kind : str
        One of ``"mode"``, ``"angle"``, ``"joint"``.
    dataset_path : str or None
        Path to the ``.pt`` dataset file.  Defaults to the configured
        ``DATASET_PATH``.

    Returns
    -------
    dict
        Keys: ``pattern``, ``pred_items``, ``best_reward``,
        ``geometry_id`` (or ``record_idx`` for mode tasks),
        ``all_rewards``, ``task_kind``.
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH

    if task_kind not in ("mode", "angle", "joint"):
        raise ValueError(
            f"Unknown task_kind: {task_kind}. "
            f"Must be one of 'mode', 'angle', 'joint'."
        )

    # Load validation set
    data_obj, _train_pairs, val_pairs = load_pair_splits(
        dataset_path,
        train_ratio=0.8,
        seed=42,
        theta_min_deg=0,
        theta_max_deg=60,
        split_by_geometry=True,
    )
    val_ds = ForwardDualModeThetaMagDataset(data_obj, val_pairs)
    freqs_ghz = _build_freqs_ghz(data_obj)
    print(
        f"  Validation samples: {len(val_ds)}, "
        f"freq points: {len(freqs_ghz)}"
    )

    # Dispatch
    if task_kind == "mode":
        pattern, pred_items, best_reward, ref, all_rewards = (
            search_dataset_mode(val_ds, freqs_ghz, fc, bw)
        )
        result = {
            "pattern": pattern,
            "pred_items": pred_items,
            "best_reward": float(best_reward),
            "record_idx": int(ref),
            "all_rewards": all_rewards,
            "task_kind": task_kind,
        }
    else:
        pattern, pred_items, best_reward, gid, all_rewards = (
            search_dataset_by_geometry(val_ds, freqs_ghz, fc, bw, task_kind)
        )
        result = {
            "pattern": pattern,
            "pred_items": pred_items,
            "best_reward": float(best_reward),
            "geometry_id": gid,
            "all_rewards": all_rewards,
            "task_kind": task_kind,
        }

    return result
