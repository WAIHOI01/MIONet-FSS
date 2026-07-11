"""Build quantitative comparison tables for inverse design results."""
import json
import os
from collections import defaultdict

import numpy as np

from common.config import STANDARD_FREQS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-8
GUARD_GHZ = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db(x):
    """Convert linear magnitude to dB."""
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


def _make_masks(n_pts, fc, bw):
    """Build in-band and out-of-band boolean masks.

    Returns ``(band_mask, out_mask, freq_array)``.
    """
    f = np.linspace(0.01, 20.0, n_pts)
    b = (f >= fc - bw / 2.0) & (f <= fc + bw / 2.0)
    lo = fc - bw / 2.0 - GUARD_GHZ
    hi = fc + bw / 2.0 + GUARD_GHZ
    o = (f < lo) | (f > hi)
    return b, o, f


def _get_s(data_dict, pol, theta_str, param):
    """Retrieve S-parameter from NPZ dict, trying pred_ then cst_ prefixes."""
    for prefix in ("pred_", "cst_"):
        key = f"{prefix}{param}_pol{pol}_theta{theta_str}"
        if key in data_dict:
            return data_dict[key]
    return None


def _align(arr, target_len):
    """Interpolate *arr* to length *target_len* if needed."""
    if len(arr) == target_len:
        return arr
    src_f = np.linspace(0.01, 20.0, len(arr))
    dst_f = np.linspace(0.01, 20.0, target_len)
    return np.interp(dst_f, src_f, arr)


def _compute_row(data_dict, fc, bw, task_kind, is_cst_ref=False):
    """Compute per-spec metrics for a single task result.

    Returns a dict of metric values.
    """
    m = {}

    # TE 0, TM 0
    s11_te = _get_s(data_dict, 0, "0", "s11")
    s21_te = _get_s(data_dict, 0, "0", "s21")
    s11_tm = _get_s(data_dict, 1, "0", "s11")
    s21_tm = _get_s(data_dict, 1, "0", "s21")

    if s21_te is not None:
        b, o, _ = _make_masks(len(s21_te), fc, bw)
        m["TE_S21_in"] = float(_db(np.mean(np.abs(s21_te[b]))))
        m["TE_S21_out"] = (
            float(_db(np.mean(np.abs(s21_te[o])))) if o.any() else float("nan")
        )
    if s11_te is not None:
        b, o, _ = _make_masks(len(s11_te), fc, bw)
        m["TE_S11_in"] = float(_db(np.mean(np.abs(s11_te[b]))))
        m["TE_S11_out"] = (
            float(_db(np.mean(np.abs(s11_te[o])))) if o.any() else float("nan")
        )

    if s21_tm is not None:
        b, o, _ = _make_masks(len(s21_tm), fc, bw)
        m["TM_S21_in"] = float(_db(np.mean(np.abs(s21_tm[b]))))
        m["TM_S21_out"] = (
            float(_db(np.mean(np.abs(s21_tm[o])))) if o.any() else float("nan")
        )
    if s11_tm is not None:
        b, o, _ = _make_masks(len(s11_tm), fc, bw)
        m["TM_S11_in"] = float(_db(np.mean(np.abs(s11_tm[b]))))
        m["TM_S11_out"] = (
            float(_db(np.mean(np.abs(s11_tm[o])))) if o.any() else float("nan")
        )

    # Contrast (in-band + out-band): difference of dB-of-means
    if s21_te is not None and s21_tm is not None:
        b, _, _ = _make_masks(len(s21_te), fc, bw)
        s21_tm_a = _align(s21_tm, len(s21_te))
        te_in_db = _db(np.mean(np.abs(s21_te[b])))
        tm_in_db = _db(np.mean(np.abs(s21_tm_a[b])))
        m["Contrast_in"] = float(te_in_db - tm_in_db)

        _, o, _ = _make_masks(len(s21_te), fc, bw)
        s21_tm_ao = _align(s21_tm, len(s21_te))
        te_out_db = (
            _db(np.mean(np.abs(s21_te[o]))) if o.any() else float("nan")
        )
        tm_out_db = (
            _db(np.mean(np.abs(s21_tm_ao[o]))) if o.any() else float("nan")
        )
        m["Contrast_out"] = float(te_out_db - tm_out_db)

    # Angular deviation
    if task_kind in ("angle", "joint"):
        s21_te60 = _get_s(data_dict, 0, "60", "s21")
        if s21_te60 is not None and s21_te is not None:
            a0 = np.abs(s21_te)
            a60 = np.abs(s21_te60)
            if len(a0) != len(a60):
                a60 = _align(a60, len(a0))
            b, _, _ = _make_masks(len(a0), fc, bw)
            m["Ang_dev"] = float(np.mean(np.abs(a0[b] - a60[b])))

    # RMSE (pred vs CST -- only applicable for RL-Pred)
    if not is_cst_ref:
        rmses = []
        for pol, theta_str in [(0, "0"), (1, "0")]:
            for param in ["s11", "s21"]:
                p = _get_s(data_dict, pol, theta_str, param)
                c_key = f"cst_{param}_pol{pol}_theta{theta_str}"
                c = data_dict.get(c_key)
                if p is not None and c is not None:
                    p_a = np.abs(_align(p, len(c)))
                    rmses.append(
                        float(np.sqrt(np.mean((p_a - np.abs(c)) ** 2)))
                    )
        if rmses:
            m["Mean_RMSE"] = float(np.mean(rmses))

    return m


# ---------------------------------------------------------------------------
# Collect results from directory structure
# ---------------------------------------------------------------------------

def collect_results(results_dirs, fc_list, bw_list):
    """Collect results from directory structure into metric rows.

    Parameters
    ----------
    results_dirs : str or dict
        Either a base directory (str) containing ``phase1_rl_design/``,
        ``phase2_dataset/``, and ``phase3_cst/`` subdirectories, or a dict
        mapping ``"phase1"``, ``"phase2"``, ``"phase3"`` to explicit paths.
    fc_list : list of float
        Centre frequencies to scan (e.g. ``[10.0, 12.0, 14.0]``).
    bw_list : list of float
        Bandwidths to scan (e.g. ``[1.0, 2.0, 3.0, 4.0]``).

    Returns
    -------
    list of dict
        Each row contains: Source, Task, fc, bw, Metal, Reward, and
        per-spec metrics (TE_S21_in, TE_S11_in, TE_S21_out, TE_S11_out,
        TM_S21_in, TM_S11_in, TM_S21_out, TM_S11_out, Contrast_in,
        Contrast_out, Ang_dev, Mean_RMSE).
    """
    if isinstance(results_dirs, str):
        base = results_dirs
        phase1 = os.path.join(base, "phase1_rl_design")
        phase2 = os.path.join(base, "phase2_dataset")
        phase3 = os.path.join(base, "phase3_cst")
    else:
        phase1 = results_dirs.get("phase1", "")
        phase2 = results_dirs.get("phase2", "")
        phase3 = results_dirs.get("phase3", "")

    rows = []

    pairings = [
        ("mode",  "mode_transmit",  "Polar-Selective"),
        ("angle", "angle_transmit", "Angle-Stable"),
        ("joint", "joint_transmit", "Joint Design"),
    ]

    for kind, rl_model, label in pairings:
        for fc in fc_list:
            for bw in bw_list:
                tid = f"fc{fc:.0f}_bw{bw:.0f}"

                # Paths
                rl_npz = os.path.join(phase1, rl_model, tid, "global_best.npz")
                rl_cst = os.path.join(phase3, kind, tid, "rl_cst.npz")
                rl_json = os.path.join(phase1, rl_model, tid, "global_best.json")
                ds_npz = os.path.join(phase2, kind, tid, "dataset_best.npz")
                ds_cst = os.path.join(phase3, kind, tid, "ds_cst.npz")
                ds_json = os.path.join(phase2, kind, tid, "dataset_best.json")

                def _ldj(p):
                    return json.load(open(p)) if os.path.exists(p) else {}

                def _ld_npz(p):
                    if os.path.exists(p):
                        return {k: np.load(p)[k] for k in np.load(p).keys()}
                    return {}

                # RL-Pred
                if os.path.exists(rl_npz):
                    data = _ld_npz(rl_npz)
                    if os.path.exists(rl_cst):
                        data.update(_ld_npz(rl_cst))
                    m = _compute_row(data, fc, bw, kind, is_cst_ref=False)
                    m["Source"] = "RL-Pred"
                    m["Task"] = label
                    m["fc"] = fc
                    m["bw"] = bw
                    m["Metal"] = _ldj(rl_json).get(
                        "n_metal",
                        int(np.load(rl_npz)["pattern_10x10"].sum()),
                    )
                    m["Reward"] = _ldj(rl_json).get("best_reward")
                    rows.append(m)

                # RL-CST
                if os.path.exists(rl_cst):
                    data = _ld_npz(rl_cst)
                    renamed = {}
                    for k, v in data.items():
                        renamed[
                            k.replace("cst_", "pred_")
                            if k.startswith("cst_")
                            else k
                        ] = v
                    m = _compute_row(renamed, fc, bw, kind, is_cst_ref=True)
                    m["Source"] = "RL-CST"
                    m["Task"] = label
                    m["fc"] = fc
                    m["bw"] = bw
                    m["Metal"] = int(np.load(rl_cst)["pattern_10x10"].sum())
                    m["Reward"] = None
                    rows.append(m)

                # DS
                if os.path.exists(ds_npz):
                    data = _ld_npz(ds_npz)
                    if os.path.exists(ds_cst):
                        data.update(_ld_npz(ds_cst))
                    m = _compute_row(data, fc, bw, kind, is_cst_ref=True)
                    m["Source"] = "DS"
                    m["Task"] = label
                    m["fc"] = fc
                    m["bw"] = bw
                    m["Metal"] = int(np.load(ds_npz)["pattern_10x10"].sum())
                    m["Reward"] = _ldj(ds_json).get("best_reward")
                    rows.append(m)

    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(v, decimals=2):
    """Format a metric value for markdown table display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{decimals}f}"


def _build_per_kind_tables(rows, title):
    """Build per-kind summary sub-tables."""
    lines = []
    lines.append(f"### {title}")
    lines.append("")

    for kind in ["Polar-Selective", "Angle-Stable", "Joint Design"]:
        lines.append(
            f"**{kind}** (n="
            f"{sum(1 for r in rows if r['Task'] == kind and r['Source'] == 'RL-Pred')}"
            f" tasks)"
        )
        lines.append("")

        if kind == "Angle-Stable":
            col_keys = [
                "TE_S21_in", "TE_S11_in", "TE_S21_out", "TE_S11_out",
                "Ang_dev", "Mean_RMSE",
            ]
            header = (
                "| Source | TE S21 in | TE S11 in | "
                "TE S21 out | TE S11 out | "
                "Ang Dev | Mean RMSE |"
            )
            sep = (
                "|--------|-----------|-----------|"
                "-------------|-------------|"
                "---------|-----------|"
            )

            def _fmt_fn(src, vals, rmse):
                return (
                    f"| {src} | {vals.get('TE_S21_in', '—')} | "
                    f"{vals.get('TE_S11_in', '—')} | "
                    f"{vals.get('TE_S21_out', '—')} | "
                    f"{vals.get('TE_S11_out', '—')} | "
                    f"{vals.get('Ang_dev', '—')} | {rmse} |"
                )
        else:
            col_keys = [
                "TE_S21_in", "TE_S11_in", "TE_S21_out", "TE_S11_out",
                "TM_S21_in", "TM_S11_in", "TM_S21_out", "TM_S11_out",
                "Contrast_in", "Contrast_out", "Ang_dev", "Mean_RMSE",
            ]
            header = (
                "| Source | TE S21 in | TE S11 in | "
                "TE S21 out | TE S11 out | "
                "TM S21 in | TM S11 in | "
                "TM S21 out | TM S11 out | "
                "Contr in | Contr out | Ang Dev | Mean RMSE |"
            )
            sep = (
                "|--------|-----------|-----------|"
                "-------------|-------------|"
                "-----------|-----------|"
                "-------------|-------------|"
                "---------|------------|---------|-----------|"
            )

            def _fmt_fn(src, vals, rmse):
                return (
                    f"| {src} | {vals.get('TE_S21_in', '—')} | "
                    f"{vals.get('TE_S11_in', '—')} | "
                    f"{vals.get('TE_S21_out', '—')} | "
                    f"{vals.get('TE_S11_out', '—')} | "
                    f"{vals.get('TM_S21_in', '—')} | "
                    f"{vals.get('TM_S11_in', '—')} | "
                    f"{vals.get('TM_S21_out', '—')} | "
                    f"{vals.get('TM_S11_out', '—')} | "
                    f"{vals.get('Contrast_in', '—')} | "
                    f"{vals.get('Contrast_out', '—')} | "
                    f"{vals.get('Ang_dev', '—')} | {rmse} |"
                )

        lines.append(header)
        lines.append(sep)

        for src in ["RL-Pred", "RL-CST", "DS"]:
            src_rows = [r for r in rows if r["Task"] == kind and r["Source"] == src]
            if not src_rows:
                continue
            vals = {}
            for k in col_keys:
                arr = [
                    r[k]
                    for r in src_rows
                    if k in r
                    and r[k] is not None
                    and not (isinstance(r[k], float) and np.isnan(r[k]))
                ]
                vals[k] = (
                    f"{np.mean(arr):.2f}±{np.std(arr):.2f}"
                    if arr
                    else "—"
                )
            rmse = (
                "—"
                if src != "RL-Pred"
                else vals.get("Mean_RMSE", "—")
            )
            lines.append(_fmt_fn(src, vals, rmse))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build full summary markdown
# ---------------------------------------------------------------------------

def build_summary_table(results_dirs, fc_list, bw_list, output_md_path):
    """Build a complete summary markdown table.

    Collects metrics from the directory structure and writes a markdown
    report to *output_md_path*.

    Parameters
    ----------
    results_dirs : str or dict
        Base directory (or dict of per-phase paths) containing results.
    fc_list : list of float
        Centre frequencies.
    bw_list : list of float
        Bandwidths.
    output_md_path : str
        Path where the markdown report will be written.

    Returns
    -------
    list of dict
        The collected metric rows (same as ``collect_results``).
    """
    rows = collect_results(results_dirs, fc_list, bw_list)

    out_dir = os.path.dirname(output_md_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as f:

        def w(s=""):
            f.write(s + "\n")

        w("# RL Inverse Design — Quantitative Evaluation")
        w()
        w("## Summary Tables")
        w()
        w(
            _build_per_kind_tables(
                rows,
                f"Summary (fc = {', '.join(f'{x:.0f}' for x in fc_list)} GHz, "
                f"{len(fc_list) * len(bw_list)} targets per task kind)",
            )
        )
        w()

        w("## Per-Task Detail")
        w()

        for kind in ["Polar-Selective", "Angle-Stable", "Joint Design"]:
            w(f"### {kind}")
            w()
            w(
                "| Source | fc | bw | TE S21 in | TE S11 in | "
                "TE S21 out | Metal | Reward |"
            )
            w(
                "|--------|----|----|-----------|-----------|"
                "-------------|-------|--------|"
            )
            kind_rows = [r for r in rows if r["Task"] == kind]
            for r in sorted(kind_rows, key=lambda x: (x["Source"], x["fc"], x["bw"])):
                w(
                    f"| {r['Source']} | {r['fc']:.0f} | {r['bw']:.0f} | "
                    f"{_fmt(r.get('TE_S21_in'))} | "
                    f"{_fmt(r.get('TE_S11_in'))} | "
                    f"{_fmt(r.get('TE_S21_out'))} | "
                    f"{r.get('Metal', '—')} | "
                    f"{_fmt(r.get('Reward'), 4)} |"
                )
            w()

        w("## RL vs Dataset Analysis")
        w()
        w(
            "Comparison of RL-CST vs DS (CST-validated). "
            "Positive delta means RL outperforms."
        )
        w()

        for kind in ["Polar-Selective", "Angle-Stable", "Joint Design"]:
            w(f"### {kind}")
            w()
            w("| Metric | RL-CST | DS | Delta |")
            w("|--------|--------|-----|-------|")
            for metric in [
                "TE_S21_in", "TE_S11_in", "TE_S21_out", "TE_S11_out",
                "TM_S21_in", "TM_S11_in", "TM_S21_out", "TM_S11_out",
                "Contrast_in", "Contrast_out", "Ang_dev",
            ]:
                rl_vals = [
                    r[metric]
                    for r in rows
                    if r["Task"] == kind
                    and r["Source"] == "RL-CST"
                    and metric in r
                    and r[metric] is not None
                    and not (isinstance(r[metric], float) and np.isnan(r[metric]))
                ]
                ds_vals = [
                    r[metric]
                    for r in rows
                    if r["Task"] == kind
                    and r["Source"] == "DS"
                    and metric in r
                    and r[metric] is not None
                    and not (isinstance(r[metric], float) and np.isnan(r[metric]))
                ]
                if rl_vals and ds_vals:
                    rl_m = np.mean(rl_vals)
                    ds_m = np.mean(ds_vals)
                    delta = rl_m - ds_m
                    w(
                        f"| {metric} | {rl_m:.2f} | {ds_m:.2f} | "
                        f"{delta:+.2f} |"
                    )
            w()

        # Save metrics JSON alongside
        json_path = output_md_path.replace(".md", ".json")
        json.dump(rows, open(json_path, "w"), indent=2, default=str)

    print(f"Summary table saved: {output_md_path}")
    return rows
