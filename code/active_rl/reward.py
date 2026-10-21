"""Reward functions: the v11 terminal S-parameter reward (canonical) and the
stability aggregation used by the environment, CST evaluation, and sweep.
"""

import numpy as np

from active_rl.config import (
    V11_GUARD_GHZ,
    V11_PASS_SCORE_TARGET,
    V11_OUT_PENALTY_TARGET,
    DEFAULT_STABILITY_WEIGHT,
)


def _mag01(values):
    return np.clip(np.abs(np.asarray(values, dtype=np.float32)), 0.0, 1.0)


def _safe_complement(mag):
    mag = _mag01(mag)
    return np.sqrt(np.maximum(0.0, 1.0 - mag * mag)).astype(np.float32)


def _band_mask(freqs, spec):
    freqs = np.asarray(freqs, dtype=np.float32)
    fc = float(spec["fc"])
    bw = float(spec["bw"])
    mask = (freqs >= fc - bw / 2.0) & (freqs <= fc + bw / 2.0)
    if not np.any(mask):
        mask = np.zeros_like(freqs, dtype=bool)
        mask[int(np.argmin(np.abs(freqs - fc)))] = True
    return mask


def _outband_mask(freqs, spec):
    freqs = np.asarray(freqs, dtype=np.float32)
    fc = float(spec["fc"])
    bw = float(spec["bw"])
    guard = max(0.0, float(spec.get("guard_ghz", V11_GUARD_GHZ)))
    lo = fc - bw / 2.0 - guard
    hi = fc + bw / 2.0 + guard
    return (freqs < lo) | (freqs > hi)

def _mean_or_zero(values):
    values = np.asarray(values, dtype=np.float32)
    return float(values.mean()) if values.size else 0.0


def _transmission_score(s11, s21):
    s11 = _mag01(s11)
    s21 = _mag01(s21)
    return 0.5 * (_mean_or_zero(s21) + _mean_or_zero(_safe_complement(s11)))


def _reflection_score(s11, s21):
    s11 = _mag01(s11)
    s21 = _mag01(s21)
    return 0.5 * (_mean_or_zero(s11) + _mean_or_zero(_safe_complement(s21)))


def calculate_sparam_reward_v11(pred_s11, pred_s21, freqs, spec):
    """Return (reward, loss_like, pass_mean) for one TE/TM specification.

    Reward is bounded approximately in [-1, 1]:
      transmit: in-band transmission score - out-band transmission penalty
      reflect : in-band reflection score   - out-band reflection penalty
    """
    s11 = _mag01(pred_s11)
    s21 = _mag01(pred_s21)
    band = _band_mask(freqs, spec)
    out = _outband_mask(freqs, spec)
    task_type = str(spec.get("type", "transmit")).lower()

    if task_type == "transmit":
        in_score = _transmission_score(s11[band], s21[band])
        out_penalty = _transmission_score(s11[out], s21[out]) if np.any(out) else 0.0
    elif task_type == "reflect":
        in_score = _reflection_score(s11[band], s21[band])
        out_penalty = _reflection_score(s11[out], s21[out]) if np.any(out) else 0.0
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    reward = float(in_score - out_penalty)
    loss_like = float((1.0 - in_score) + out_penalty)
    pass_mean = bool(
        in_score >= V11_PASS_SCORE_TARGET and
        out_penalty <= V11_OUT_PENALTY_TARGET
    )
    return reward, loss_like, pass_mean


def task_reward_v11(pred_s11, pred_s21, freqs, spec):
    return calculate_sparam_reward_v11(pred_s11, pred_s21, freqs, spec)


def aggregate_stability_reward(rewards, specs=None, stability_weight=DEFAULT_STABILITY_WEIGHT):
    values = np.asarray(rewards, dtype=np.float32)
    if values.size == 0:
        return 0.0
    mean_reward = float(values.mean())
    if stability_weight <= 0 or values.size <= 1:
        return mean_reward

    if specs is None:
        stability_penalty = float(values.std())
    else:
        grouped = {}
        for reward, spec in zip(values, specs):
            key = (int(spec.get("pol", 0)), str(spec.get("type", "")))
            grouped.setdefault(key, []).append(float(reward))
        stds = [float(np.asarray(group, dtype=np.float32).std())
                for group in grouped.values() if len(group) > 1]
        stability_penalty = float(np.mean(stds)) if stds else 0.0
    return mean_reward - float(stability_weight) * stability_penalty
