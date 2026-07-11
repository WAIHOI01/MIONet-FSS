"""
Paper S-parameter figures — one fc per task kind, no title
============================================================
mode   fc10_bw1: TE + TM (2 figs)
angle  fc12_bw2: 0° + 60° (2 figs)
joint  fc14_bw4: TE 0°/60° + TM 0°/60° (4 figs)

Reads data from ../code/results_benchmark_16tasks/ and outputs figures
to ../code/results_figures/paper_sparams/.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_S11 = "#1F6DB3"
C_S21 = "#C0182F"
C_DS_S11 = "#2E7D32"
C_DS_S21 = "#E65100"
C_BAND = "#81C784"
EPS = 1e-8
FIG_W = 3.5
FIG_H = 2.65

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 6.5, "axes.linewidth": 0.8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "..", "code", "results_benchmark_16tasks")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "code", "results_figures", "paper_sparams")
os.makedirs(OUT_DIR, exist_ok=True)

CASES = [
    # (kind, rl_model, ds_dir, fc, bw, panels)
    # panel = (label, pol, theta)
    ("mode",  "mode_transmit",  "mode",  10, 1, [("TE", 0, 0), ("TM", 1, 0)]),
    ("angle", "angle_transmit", "angle", 12, 2, [("TE_0deg", 0, 0), ("TE_60deg", 0, 60)]),
    ("joint", "joint_transmit", "joint", 14, 4, [
        ("TE_0deg", 0, 0), ("TE_60deg", 0, 60),
        ("TM_0deg", 1, 0), ("TM_60deg", 1, 60),
    ]),
]


def load_one(kind, rl_model, ds_dir, fc, bw):
    tid = f"fc{fc:.0f}_bw{bw:.0f}"
    data = {}
    for tag, path, pfx in [
        ("pred", os.path.join(BASE, "phase1_rl_design", rl_model, tid, "global_best.npz"), "pred_"),
        ("cst",  os.path.join(BASE, "phase3_cst", kind, tid, "rl_cst.npz"), "cst_"),
        ("ds",   os.path.join(BASE, "phase2_dataset", ds_dir, tid, "dataset_best.npz"), "pred_"),
    ]:
        if not os.path.exists(path):
            continue
        d = np.load(path)
        data[tag] = {"f": d["freqs_ghz"],
                     "c": {k: d[k] for k in d.keys() if k.startswith(pfx)}}
    return data


def get_s(curves, pol, theta, param):
    for t in [str(int(theta))]:
        for pfx in [f"pred_s{param}", f"cst_s{param}"]:
            k = f"{pfx}_pol{pol}_theta{t}"
            if k in curves:
                return curves[k]
    return None


def align(arr, sf, tf):
    if arr is None or len(arr) == len(tf):
        return arr
    return np.interp(tf, sf, arr)


def draw_pair(ax, f_ref, s11, s21, cs11, cs21, ls, mk, ms, alpha,
              l11, l21):
    if s11 is not None:
        ax.plot(f_ref, 20 * np.log10(np.abs(s11) + EPS),
                color=cs11, lw=1.1, ls=ls, marker=mk, ms=ms,
                markevery=25, markerfacecolor=cs11,
                markeredgecolor="white", markeredgewidth=0.4,
                alpha=alpha, label=l11, zorder=3)
    if s21 is not None:
        ax.plot(f_ref, 20 * np.log10(np.abs(s21) + EPS),
                color=cs21, lw=1.1, ls=ls, marker=mk, ms=ms,
                markevery=25, markerfacecolor=cs21,
                markeredgecolor="white", markeredgewidth=0.4,
                alpha=alpha, label=l21, zorder=3)


def main():
    for kind, rm, ds_dir, fc, bw, panels in CASES:
        data = load_one(kind, rm, ds_dir, fc, bw)
        f_ref = data.get("cst", data.get("pred", {})).get("f")
        if f_ref is None:
            continue

        for plabel, pol, theta in panels:
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

            # RL-Pred
            if "pred" in data:
                s11 = align(get_s(data["pred"]["c"], pol, theta, "11"),
                            data["pred"]["f"], f_ref)
                s21 = align(get_s(data["pred"]["c"], pol, theta, "21"),
                            data["pred"]["f"], f_ref)
                draw_pair(ax, f_ref, s11, s21, C_S11, C_S21,
                          "-", "o", 4.0, 0.95,
                          r"$S_{11}$ Pred", r"$S_{21}$ Pred")

            # RL-CST
            if "cst" in data:
                s11 = get_s(data["cst"]["c"], pol, theta, "11")
                s21 = get_s(data["cst"]["c"], pol, theta, "21")
                draw_pair(ax, f_ref, s11, s21, C_S11, C_S21,
                          "--", "", 0, 0.52,
                          r"$S_{11}$ CST", r"$S_{21}$ CST")

            # DS-Best
            if "ds" in data:
                s11 = align(get_s(data["ds"]["c"], pol, theta, "11"),
                            data["ds"]["f"], f_ref)
                s21 = align(get_s(data["ds"]["c"], pol, theta, "21"),
                            data["ds"]["f"], f_ref)
                draw_pair(ax, f_ref, s11, s21, C_DS_S11, C_DS_S21,
                          "-", "s", 2.5, 0.82,
                          r"$S_{11}$ DS-Best", r"$S_{21}$ DS-Best")

            # Target band
            lo, hi = fc - bw / 2, fc + bw / 2
            ax.axvspan(lo, hi, alpha=0.22, color=C_BAND)
            for b in (lo, hi):
                ax.axvline(b, color=C_BAND, lw=0.6, ls=":", alpha=0.5)

            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(r"$S$-parameter (dB)")
            ax.set_ylim(-50, 5)
            ax.set_xlim(0, 20)
            ax.grid(True, alpha=0.28)

            ax.legend(loc="lower left", frameon=True, fancybox=False,
                      framealpha=0.50, edgecolor="#cccccc",
                      fontsize=7, ncol=3, columnspacing=0.4,
                      handlelength=1.4, handletextpad=0.3)

            fig.tight_layout(pad=0.3)
            fname = f"{kind}_fc{fc:.0f}_bw{bw:.0f}_{plabel}"
            for ext in [".png", ".pdf"]:
                fig.savefig(os.path.join(OUT_DIR, fname + ext),
                            dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fname}")

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
