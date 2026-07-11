"""Task specification builders for FSS inverse design."""
import numpy as np
from common.config import STANDARD_FREQS, N_FREQ

ANGLES_DEG = [0, 10, 20, 30, 40, 50, 60]


def build_target_masks(freqs, fc, bw, design_type):
    """Build S11/S21 target masks based on design type (transmit/reflect).

    Parameters
    ----------
    freqs : ndarray
        Frequency grid (N_FREQ,).
    fc : float
        Center frequency in GHz.
    bw : float
        Bandwidth in GHz.
    design_type : str
        ``"transmit"`` or ``"reflect"``.

    Returns
    -------
    target_s11 : ndarray
        Ideal |S11| target at each frequency.
    target_s21 : ndarray
        Ideal |S21| target at each frequency.
    """
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


def opposite_design_type(t):
    """Return the opposite design type."""
    if t == "transmit":
        return "reflect"
    if t == "reflect":
        return "transmit"
    raise ValueError(f"Unknown design_type: {t}")


def normalize_pol(pol):
    """Normalize polarization to 0 (TE) or 1 (TM)."""
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
    """Return 'TE' or 'TM' string for polarization integer."""
    return "TE" if normalize_pol(pol) == 0 else "TM"


def build_empty_target_masks():
    """Return a dict of zero-filled target masks for both polarizations."""
    z = np.zeros(N_FREQ, dtype=np.float32)
    return {"te_s11": z.copy(), "te_s21": z.copy(), "tm_s11": z.copy(), "tm_s21": z.copy()}


def build_polarization_selection_task(fc, bw, te_design_type, theta=0.0):
    """2 specs: TE transmit + TM reflect at a fixed incident angle.

    Parameters
    ----------
    fc : float
        Center frequency in GHz.
    bw : float
        Bandwidth in GHz.
    te_design_type : str
        ``"transmit"`` — TE is pass-band, TM is stop-band.
        ``"reflect"``  — TE is stop-band, TM is pass-band.
    theta : float
        Incident angle in degrees (default 0).

    Returns
    -------
    specs : list of dict
        Two specification dicts (TE + TM).
    masks : dict
        Target masks keyed by ``"te_s11"``, ``"te_s21"``, ``"tm_s11"``, ``"tm_s21"``.
    """
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


def build_angle_stability_task(fc, bw, design_type, pol=0):
    """7 specs: single polarization across 0-60 degrees.

    Parameters
    ----------
    fc : float
        Center frequency in GHz.
    bw : float
        Bandwidth in GHz.
    design_type : str
        ``"transmit"`` or ``"reflect"``.
    pol : int or str
        Polarization (0 / ``"TE"`` or 1 / ``"TM"``).

    Returns
    -------
    specs : list of dict
        Seven specification dicts, one per angle.
    masks : dict
        Target masks with only the active polarization populated; the other
        polarization entries are zero-filled.
    """
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


def build_joint_angle_stability_task(fc, bw, te_design_type):
    """14 specs: TE+TM across 0-60 degrees with opposite pass/stop functions.

    TE and TM share the same fc/bw but have complementary design types.

    Parameters
    ----------
    fc : float
        Center frequency in GHz.
    bw : float
        Bandwidth in GHz.
    te_design_type : str
        ``"transmit"`` or ``"reflect"`` for the TE polarization.

    Returns
    -------
    specs : list of dict
        Fourteen specification dicts (7 angles x 2 polarizations).
    masks : dict
        Target masks keyed by ``"te_s11"``, ``"te_s21"``, ``"tm_s11"``, ``"tm_s21"``.
    """
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


def build_active_design_task(fc, bw, task_kind, te_design_type, angle_pol=0):
    """Dispatch to the correct task builder based on *task_kind*.

    Parameters
    ----------
    fc : float
        Center frequency in GHz.
    bw : float
        Bandwidth in GHz.
    task_kind : str
        One of ``"mode"``, ``"angle"``, ``"joint"``.
    te_design_type : str
        ``"transmit"`` or ``"reflect"`` — used by ``"mode"`` and ``"joint"``.
    angle_pol : int or str
        Polarization for ``"angle"`` tasks (0 / ``"TE"`` or 1 / ``"TM"``).

    Returns
    -------
    specs : list of dict
    masks : dict
    """
    if task_kind == "joint":
        return build_joint_angle_stability_task(fc, bw, te_design_type)
    if task_kind == "angle":
        return build_angle_stability_task(fc, bw, te_design_type, pol=angle_pol)
    if task_kind == "mode":
        return build_polarization_selection_task(fc, bw, te_design_type, theta=0.0)
    raise ValueError(f"Unknown task_kind: {task_kind}")
