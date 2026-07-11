"""v11 reward function for FSS inverse design — self-contained, no version inheritance."""
import numpy as np

V11_GUARD_GHZ = 0.25
V11_PASS_SCORE_TARGET = 0.89
V11_OUT_PENALTY_TARGET = 0.31
DEFAULT_STABILITY_WEIGHT = 0.25


def _mag01(x):
    """Clip magnitude to [0, 1]."""
    return np.clip(np.abs(np.asarray(x, dtype=np.float32)), 0.0, 1.0)


def _band_mask(freqs, spec):
    """Boolean mask for in-band frequencies."""
    freqs = np.asarray(freqs, dtype=np.float32)
    fc = float(spec["fc"])
    bw = float(spec["bw"])
    return (freqs >= fc - bw / 2.0) & (freqs <= fc + bw / 2.0)


def _outband_mask(freqs, spec):
    """Boolean mask for out-of-band frequencies (with guard gap)."""
    freqs = np.asarray(freqs, dtype=np.float32)
    fc = float(spec["fc"])
    bw = float(spec["bw"])
    guard = max(0.0, float(spec.get("guard_ghz", V11_GUARD_GHZ)))
    lo = fc - bw / 2.0 - guard
    hi = fc + bw / 2.0 + guard
    return (freqs < lo) | (freqs > hi)


def _transmission_score(s11_mag, s21_mag):
    """In-band transmission quality: average of |S21| and sqrt(1-|S11|^2)."""
    s11 = np.asarray(s11_mag, dtype=np.float32)
    s21 = np.asarray(s21_mag, dtype=np.float32)
    return float(np.mean(0.5 * (s21 + np.sqrt(np.maximum(1.0 - s11**2, 0.0)))))


def _reflection_score(s11_mag, s21_mag):
    """In-band reflection quality: average of |S11| and sqrt(1-|S21|^2)."""
    s11 = np.asarray(s11_mag, dtype=np.float32)
    s21 = np.asarray(s21_mag, dtype=np.float32)
    return float(np.mean(0.5 * (s11 + np.sqrt(np.maximum(1.0 - s21**2, 0.0)))))


def calculate_sparam_reward_v11(pred_s11, pred_s21, freqs, spec):
    """Return (reward, loss_like, pass_mean) for one TE/TM specification.

    Reward ≈ in_score - out_penalty, bounded roughly in [-1, 1].
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


def aggregate_stability_reward(rewards, specs=None, stability_weight=DEFAULT_STABILITY_WEIGHT):
    """Aggregate per-spec rewards with stability penalty.

    Returns mean_reward - stability_weight * stability_penalty,
    where stability_penalty is the mean of per-group standard deviations.
    """
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
