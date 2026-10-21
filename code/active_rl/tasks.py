"""Task-spec builders (mode / angle / joint) and the v11 active-design sampler.

Module globals TARGET_FC_NOISE / TARGET_BW_NOISE / TARGET_SAMPLING are set by
the trainer from CLI args before the environment is constructed (same order as
the original v11.train()).
"""

import random

import numpy as np

from active_rl.config import (
    ANGLES_DEG,
    FEASIBLE_FC_RANGE,
    FEASIBLE_BW_RANGE,
    SWEEP_DESIGN_TYPES,
    STANDARD_FREQS,
    N_FREQ,
)

V11_FC_LIST = [8.0, 10.0, 12.0, 14.0]
V11_BW_LIST = [1.0, 2.0, 3.0, 4.0]
V11_TARGET_KEYS = ("te_target_s11", "te_target_s21", "tm_target_s11", "tm_target_s21")
V11_CURVE_ANGLES = (0.0, 30.0, 60.0)


def v11_angle_tag(theta):
    return f"{float(theta):g}".replace(".", "p").replace("-", "m")


def v11_curve_key(pol, param, theta):
    pol_name = "te" if int(pol) == 0 else "tm"
    return f"cur_{pol_name}_{str(param).lower()}_t{v11_angle_tag(theta)}"


V11_CURVE_KEYS = tuple(
    v11_curve_key(pol, param, theta)
    for theta in V11_CURVE_ANGLES
    for pol in (0, 1)
    for param in ("s11", "s21")
)

TARGET_FC_NOISE = 0.0
TARGET_BW_NOISE = 0.0
TARGET_SAMPLING = "grid"  # "grid" | "continuous" (v9-style uniform)


# ============================================================
#  Joint TE/TM angle-stability target model
# ============================================================
def build_target_masks(freqs, fc, bw, design_type):
    target_s11 = np.ones_like(freqs, dtype=np.float32)
    target_s21 = np.zeros_like(freqs, dtype=np.float32)
    band_idx = (freqs >= (fc - bw / 2.0)) & (freqs <= (fc + bw / 2.0))

    if design_type == "transmit":
        target_s11[band_idx] = 0.0
        target_s21[band_idx] = 1.0
    elif design_type == "reflect":
        target_s11[:] = 0.0
        target_s21[:] = 1.0
        target_s11[band_idx] = 1.0
        target_s21[band_idx] = 0.0
    else:
        raise ValueError(f"Unknown design_type: {design_type}")
    return target_s11, target_s21


def opposite_design_type(design_type):
    if design_type == "transmit":
        return "reflect"
    if design_type == "reflect":
        return "transmit"
    raise ValueError(f"Unknown design_type: {design_type}")


def build_joint_angle_stability_task(fc, bw, te_design_type):
    """Build TE/TM-opposite specs at identical fc/bw for every angle."""
    if te_design_type not in ("transmit", "reflect"):
        raise ValueError(f"Unsupported TE design type: {te_design_type}")

    tm_design_type = opposite_design_type(te_design_type)
    specs = []
    for theta in ANGLES_DEG:
        common = {
            "theta": float(theta),
            "fc": float(fc),
            "bw": float(bw),
            "il_target": -3.0,
            "rl_target": -8.0,
        }
        specs.append({
            "type": te_design_type,
            "pol": 0,
            "label": f"TE_{te_design_type}@{theta}deg",
            **common,
        })
        specs.append({
            "type": tm_design_type,
            "pol": 1,
            "label": f"TM_{tm_design_type}@{theta}deg",
            **common,
        })

    te_s11, te_s21 = build_target_masks(STANDARD_FREQS, fc, bw, te_design_type)
    tm_s11, tm_s21 = build_target_masks(STANDARD_FREQS, fc, bw, tm_design_type)
    masks = {"te_s11": te_s11, "te_s21": te_s21, "tm_s11": tm_s11, "tm_s21": tm_s21}
    return specs, masks


def normalize_pol(pol):
    if isinstance(pol, str):
        key = pol.strip().lower()
        if key == "te":
            return 0
        if key == "tm":
            return 1
    if int(pol) in (0, 1):
        return int(pol)
    raise ValueError(f"Unknown polarization: {pol}")


def pol_label(pol):
    return "TE" if normalize_pol(pol) == 0 else "TM"


def build_empty_target_masks():
    z = np.zeros(N_FREQ, dtype=np.float32)
    return {"te_s11": z.copy(), "te_s21": z.copy(), "tm_s11": z.copy(), "tm_s21": z.copy()}


def build_angle_stability_task(fc, bw, design_type, pol=0):
    """Build a single-polarization task evaluated across all incident angles."""
    if design_type not in ("transmit", "reflect"):
        raise ValueError(f"Unsupported design type: {design_type}")
    pol = normalize_pol(pol)
    plabel = pol_label(pol)
    specs = []
    for theta in ANGLES_DEG:
        specs.append({
            "type": design_type,
            "pol": pol,
            "theta": float(theta),
            "fc": float(fc),
            "bw": float(bw),
            "il_target": -3.0,
            "rl_target": -8.0,
            "label": f"{plabel}_{design_type}@{theta}deg",
        })

    s11, s21 = build_target_masks(STANDARD_FREQS, fc, bw, design_type)
    masks = build_empty_target_masks()
    prefix = "te" if pol == 0 else "tm"
    masks[f"{prefix}_s11"] = s11
    masks[f"{prefix}_s21"] = s21
    return specs, masks


def build_polarization_selection_task(fc, bw, te_design_type, theta=0.0):
    """Build a theta-fixed TE/TM task with opposite pass/stop functions."""
    if te_design_type not in ("transmit", "reflect"):
        raise ValueError(f"Unsupported TE design type: {te_design_type}")
    tm_design_type = opposite_design_type(te_design_type)
    common = {
        "theta": float(theta),
        "fc": float(fc),
        "bw": float(bw),
        "il_target": -3.0,
        "rl_target": -8.0,
    }
    specs = [
        {"type": te_design_type, "pol": 0, "label": f"TE_{te_design_type}@{float(theta):g}deg", **common},
        {"type": tm_design_type, "pol": 1, "label": f"TM_{tm_design_type}@{float(theta):g}deg", **common},
    ]
    te_s11, te_s21 = build_target_masks(STANDARD_FREQS, fc, bw, te_design_type)
    tm_s11, tm_s21 = build_target_masks(STANDARD_FREQS, fc, bw, tm_design_type)
    masks = {"te_s11": te_s11, "te_s21": te_s21, "tm_s11": tm_s11, "tm_s21": tm_s21}
    return specs, masks


def build_active_design_task(fc, bw, task_kind, te_design_type, angle_pol=0):
    if task_kind == "joint":
        return build_joint_angle_stability_task(fc, bw, te_design_type)
    if task_kind == "angle":
        return build_angle_stability_task(fc, bw, te_design_type, pol=angle_pol)
    if task_kind == "mode":
        return build_polarization_selection_task(fc, bw, te_design_type, theta=0.0)
    raise ValueError(f"Unknown task_kind: {task_kind}")


def _clip_target(fc, bw):
    fc = float(np.clip(fc, FEASIBLE_FC_RANGE[0], FEASIBLE_FC_RANGE[1]))
    bw = float(np.clip(bw, FEASIBLE_BW_RANGE[0], FEASIBLE_BW_RANGE[1]))
    return fc, bw


def sample_active_design_task(freqs, task_kind="joint", design_mode="both",
                              angle_pol=0, fc_noise=None, bw_noise=None,
                              sampling=None):
    del freqs
    sampling = TARGET_SAMPLING if sampling is None else str(sampling)
    if sampling == "continuous":
        # v9-style uniform sampling (matches paper run 183059); the
        # grid-centered noise perturbation below does not apply here.
        fc = random.uniform(*FEASIBLE_FC_RANGE)
        bw = random.uniform(*FEASIBLE_BW_RANGE)
        fc, bw = _clip_target(fc, bw)
    else:
        fc = float(random.choice(V11_FC_LIST))
        bw = float(random.choice(V11_BW_LIST))
        fc_noise = TARGET_FC_NOISE if fc_noise is None else float(fc_noise)
        bw_noise = TARGET_BW_NOISE if bw_noise is None else float(bw_noise)
        if fc_noise > 0.0:
            fc += random.uniform(-fc_noise, fc_noise)
        if bw_noise > 0.0:
            bw += random.uniform(-bw_noise, bw_noise)
        fc, bw = _clip_target(fc, bw)

    if design_mode == "both":
        te_design_type = random.choice(SWEEP_DESIGN_TYPES)
    elif design_mode in SWEEP_DESIGN_TYPES:
        te_design_type = design_mode
    else:
        raise ValueError(f"Unknown design_mode: {design_mode}")

    specs, masks = build_active_design_task(
        fc, bw, task_kind, te_design_type, angle_pol=angle_pol)
    return specs, masks, fc, bw, te_design_type
