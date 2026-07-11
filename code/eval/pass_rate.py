"""Pass rate computation using physical dB thresholds."""
import os

import numpy as np

from common.config import STANDARD_FREQS
from inverse.task import build_active_design_task

# ---------------------------------------------------------------------------
# Physical dB thresholds
# ---------------------------------------------------------------------------
S21_IN_DB = -3.0   # in-band insertion loss >= -3 dB  (transmit)
S11_IN_DB = -8.0   # in-band return loss <= -8 dB
S21_OUT_DB = -8.0  # out-band rejection >= 8 dB
S11_OUT_DB = -8.0
GUARD_GHZ = 0.25

# Linear magnitude equivalents
S21_IN_LIN = 10 ** (-3.0 / 20)   # 0.7079
S11_IN_LIN = 10 ** (-8.0 / 20)   # 0.3981
S21_OUT_LIN = 10 ** (-8.0 / 20)  # 0.3981
S11_OUT_LIN = 10 ** (-8.0 / 20)  # 0.3981


# ---------------------------------------------------------------------------
# Single-spec check
# ---------------------------------------------------------------------------

def check_spec_pass(s11_mag, s21_mag, spec):
    """Check if a single spec passes physical dB thresholds.

    Criteria
    --------
    Transmit:
        S21_in  >= -3 dB   (|S21| averaged over in-band)
        S11_in  <= -8 dB   (|S11| averaged over in-band)
        S21_out <= -8 dB   (|S21| averaged over out-of-band)

    Reflect:
        S11_in  >= -3 dB   (|S11| averaged over in-band)
        S21_in  <= -8 dB   (|S21| averaged over in-band)
        S11_out <= -8 dB   (|S11| averaged over out-of-band)

    Parameters
    ----------
    s11_mag : ndarray
        Linear magnitude of |S11|, ``(N_FREQ,)`` or broadcastable.
    s21_mag : ndarray
        Linear magnitude of |S21|, ``(N_FREQ,)`` or broadcastable.
    spec : dict
        Specification with keys ``type``, ``fc``, ``bw``.

    Returns
    -------
    bool
    """
    f = STANDARD_FREQS
    fc = float(spec["fc"])
    bw = float(spec["bw"])

    # In-band mask
    band = (f >= fc - bw / 2.0) & (f <= fc + bw / 2.0)
    # Out-of-band mask (with guard gap)
    lo = fc - bw / 2.0 - GUARD_GHZ
    hi = fc + bw / 2.0 + GUARD_GHZ
    out = (f < lo) | (f > hi)

    s11_b = np.mean(np.abs(s11_mag[band]))
    s21_b = np.mean(np.abs(s21_mag[band]))
    s11_o = np.mean(np.abs(s11_mag[out])) if out.any() else 0.0
    s21_o = np.mean(np.abs(s21_mag[out])) if out.any() else 0.0

    spec_type = spec.get("type", "transmit")
    if spec_type == "transmit":
        in_ok = (s21_b >= S21_IN_LIN) and (s11_b <= S11_IN_LIN)
        out_ok = (s21_o <= S21_OUT_LIN)
    else:  # reflect
        in_ok = (s11_b >= S21_IN_LIN) and (s21_b <= S11_IN_LIN)
        out_ok = (s11_o <= S11_OUT_LIN)

    return bool(in_ok and out_ok)


# ---------------------------------------------------------------------------
# Batch pass-rate computation
# ---------------------------------------------------------------------------

def _get_s(data_dict, pol, theta_val, param):
    """Retrieve S-parameter array from an NPZ dict.

    Tries ``pred_*`` then ``cst_*`` key prefixes.
    """
    for prefix in ("pred_", "cst_"):
        key = f"{prefix}{param}_pol{pol}_theta{theta_val}"
        if key in data_dict:
            return data_dict[key]
    return None


def _align_to_standard(arr):
    """Interpolate *arr* onto ``STANDARD_FREQS`` if lengths differ."""
    if arr is None:
        return None
    if len(arr) == len(STANDARD_FREQS):
        return arr
    src_f = np.linspace(0.01, 20.0, len(arr))
    return np.interp(STANDARD_FREQS, src_f, arr)


def _task_key(fc, bw, kind):
    """Canonical key for a fc/bw/kind combination."""
    return f"fc{float(fc):.0f}_bw{float(bw):.0f}_{kind}"


def compute_pass_rates(results_by_method, fc_list=None, bw_list=None,
                       task_kinds=None):
    """Compute spec-level and task-level pass rates for all methods.

    Parameters
    ----------
    results_by_method : dict
        Maps method name (e.g. ``"RL"``, ``"GA"``, ``"Random"``, ``"DS"``)
        to a list of per-task result dicts.  Each result dict must contain
        at least:

        - ``task_kind`` : str  -- one of ``"mode"``, ``"angle"``, ``"joint"``
        - ``fc`` : float
        - ``bw`` : float
        - ``npz_path`` : str  -- path to the ``.npz`` file with S-parameter
          arrays (keys like ``pred_s11_pol0_theta0``).

    fc_list : list of float, optional
        If given, only these centre frequencies are evaluated.
    bw_list : list of float, optional
        If given, only these bandwidths are evaluated.
    task_kinds : list of str, optional
        If given, only these task kinds are evaluated.

    Returns
    -------
    dict
        Nested dict::

            {
                "<method>": {
                    "<kind>": [
                        {
                            "tid": "fc10_bw1",
                            "fc": 10.0,
                            "bw": 1.0,
                            "n_pass": 5,
                            "n_total": 7,
                            "all_passed": True,
                        },
                        ...
                    ],
                    ...
                },
                ...
            }
    """
    if task_kinds is None:
        task_kinds = ["mode", "angle", "joint"]
    if fc_list is None:
        fc_list = [10.0, 12.0, 14.0]
    if bw_list is None:
        bw_list = [1.0, 2.0, 3.0, 4.0]

    all_results = {}

    for method_name, task_list in results_by_method.items():
        for entry in task_list:
            kind = entry.get("task_kind", "")
            fc = float(entry.get("fc", 0))
            bw = float(entry.get("bw", 0))
            npz_path = entry.get("npz_path", "")

            if kind not in task_kinds:
                continue
            if fc_list and fc not in fc_list:
                continue
            if bw_list and bw not in bw_list:
                continue
            if not npz_path or not os.path.exists(npz_path):
                continue

            key = f"{method_name}/{kind}"
            if key not in all_results:
                all_results[key] = []

            # Load NPZ and build specs for this task
            d = np.load(npz_path)
            specs, _ = build_active_design_task(fc, bw, kind, "transmit", angle_pol=0)

            spec_passes = []
            for s in specs:
                pol = int(s["pol"])
                theta_val = int(s["theta"])

                s11 = _get_s(d, pol, theta_val, "s11")
                s21 = _get_s(d, pol, theta_val, "s21")

                if s11 is None or s21 is None:
                    continue

                s11 = _align_to_standard(s11)
                s21 = _align_to_standard(s21)

                spec_passes.append(check_spec_pass(np.abs(s11), np.abs(s21), s))

            tid = f"fc{fc:.0f}_bw{bw:.0f}"
            all_results[key].append({
                "tid": tid,
                "fc": fc,
                "bw": bw,
                "n_pass": sum(spec_passes),
                "n_total": len(spec_passes),
                "all_passed": all(spec_passes) if spec_passes else False,
            })

    return all_results
