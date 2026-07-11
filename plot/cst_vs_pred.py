"""
CST vs RL-prediction comparison for TE and TM polarizations.

Usage:
  python cst_vs_pred.py <pred_npz> <cst_npz> --fc 10 --bw 1

Reads:
  pred_npz — RL prediction NPZ (contains pred_s11_*, pred_s21_*, freqs_ghz, pattern_10x10)
  cst_npz  — CST simulation NPZ (contains cst_s11_*, cst_s21_*, freqs_ghz)

Output: global_best_cst_comparison_TE.png and _TM.png in the pred_npz directory.
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Single-column style -------------------------------------------
FIG_W = 3.5
FIG_H = 2.65

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.8,
})

EPS = 1e-8
C_S11 = "#1F6DB3"
C_S21 = "#C0182F"
N_FREQ = 1001


def run_comparison(pred_npz, cst_npz, fc, bw):
    pred_data = np.load(pred_npz)
    cst_data = np.load(cst_npz)

    pattern = pred_data.get("pattern_10x10",
                            cst_data.get("pattern_10x10", None))
    freqs_pred = pred_data["freqs_ghz"]
    freqs_cst = cst_data["freqs_ghz"]

    # Extract pred S-params
    te_s11_pred = pred_data["pred_s11_pol0_theta0"]
    te_s21_pred = pred_data["pred_s21_pol0_theta0"]
    tm_s11_pred = pred_data["pred_s11_pol1_theta0"]
    tm_s21_pred = pred_data["pred_s21_pol1_theta0"]

    # Extract CST S-params
    te_s11_cst = np.abs(cst_data.get("cst_s11_pol0_theta0",
                                       cst_data.get("pred_s11_pol0_theta0")))
    te_s21_cst = np.abs(cst_data.get("cst_s21_pol0_theta0",
                                       cst_data.get("pred_s21_pol0_theta0")))
    tm_s11_cst = np.abs(cst_data.get("cst_s11_pol1_theta0",
                                       cst_data.get("pred_s11_pol1_theta0")))
    tm_s21_cst = np.abs(cst_data.get("cst_s21_pol1_theta0",
                                       cst_data.get("pred_s21_pol1_theta0")))

    n_metal = int(pattern.sum()) if pattern is not None else 0
    print(f"Pattern: {n_metal}/100 metal, Target: fc={fc} bw={bw}")

    # Frequency alignment
    f_plot = freqs_cst
    n_cst = len(te_s11_cst)
    if n_cst != len(freqs_pred):
        interp = lambda pred: np.interp(freqs_cst, freqs_pred, pred)
        te_s11_pred = interp(te_s11_pred)
        te_s21_pred = interp(te_s21_pred)
        tm_s11_pred = interp(tm_s11_pred)
        tm_s21_pred = interp(tm_s21_pred)

    band = (fc - bw / 2, fc + bw / 2)
    pairs = [("TE", te_s11_pred, te_s21_pred, te_s11_cst, te_s21_cst),
             ("TM", tm_s11_pred, tm_s21_pred, tm_s11_cst, tm_s21_cst)]

    out_dir = os.path.dirname(cst_npz)

    for pol_name, s11_pred, s21_pred, s11_cst, s21_cst in pairs:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        # S11
        ax.plot(f_plot, 20 * np.log10(np.abs(s11_pred) + EPS),
                color=C_S11, lw=1.10, ls="-", marker="o", markersize=3.2,
                markevery=20, markerfacecolor=C_S11,
                markeredgecolor="white", markeredgewidth=0.55,
                alpha=0.98, label=r"$S_{11}$ Pred", zorder=3)
        ax.plot(f_plot, 20 * np.log10(np.abs(s11_cst) + EPS),
                color=C_S11, lw=1.35, ls="--", alpha=0.55,
                label=r"$S_{11}$ CST", zorder=2)

        # S21
        ax.plot(f_plot, 20 * np.log10(np.abs(s21_pred) + EPS),
                color=C_S21, lw=1.10, ls="-", marker="o", markersize=3.2,
                markevery=20, markerfacecolor=C_S21,
                markeredgecolor="white", markeredgewidth=0.55,
                alpha=0.98, label=r"$S_{21}$ Pred", zorder=3)
        ax.plot(f_plot, 20 * np.log10(np.abs(s21_cst) + EPS),
                color=C_S21, lw=1.35, ls="--", alpha=0.55,
                label=r"$S_{21}$ CST", zorder=2)

        # Target band
        ax.axvspan(*band, alpha=0.10, color="#4CAF50")
        for b in band:
            ax.axvline(b, color="#4CAF50", lw=0.6, ls=":", alpha=0.6)

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(r"$S$-parameter (dB)")
        ax.set_ylim(-50, 5)
        ax.set_xlim(0, 20)
        ax.grid(True, alpha=0.3)

        ax.legend(loc="lower left", frameon=True, fancybox=False,
                  framealpha=0.55, edgecolor="#cccccc",
                  fontsize=6.5, ncol=2,
                  columnspacing=0.5, handlelength=1.6,
                  handletextpad=0.4)

        # Pattern inset
        if pattern is not None:
            ax_ins = ax.inset_axes([0.75, 0.03, 0.25, 0.25])
            ax_ins.imshow(pattern.T, cmap="binary", origin="lower",
                          vmin=0, vmax=1, interpolation="nearest")
            ax_ins.set_xticks([])
            ax_ins.set_yticks([])
            for spine in ax_ins.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#aaaaaa")

        fig.tight_layout(pad=0.3)
        out = os.path.join(out_dir, f"global_best_cst_comparison_{pol_name}.png")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        fig.savefig(out.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    _print_errors("TE", te_s11_pred, te_s21_pred, te_s11_cst, te_s21_cst)
    _print_errors("TM", tm_s11_pred, tm_s21_pred, tm_s11_cst, tm_s21_cst)


def _print_errors(label, s11_pred, s21_pred, s11_cst, s21_cst):
    for name, p, c in [(f"{label} S11", s11_pred, s11_cst),
                        (f"{label} S21", s21_pred, s21_cst)]:
        p_a, c_a = np.abs(p), np.abs(c)
        print(f"  {name}: MAE={np.mean(np.abs(p_a - c_a)):.4f}  "
              f"RMSE={np.sqrt(np.mean((p_a - c_a)**2)):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL-predicted S-params with CST simulation.")
    parser.add_argument("pred_npz", help="Path to prediction NPZ (e.g. global_best.npz)")
    parser.add_argument("cst_npz", help="Path to CST simulation NPZ (e.g. rl_cst.npz)")
    parser.add_argument("--fc", type=float, default=10.0)
    parser.add_argument("--bw", type=float, default=1.0)
    args = parser.parse_args()
    run_comparison(args.pred_npz, args.cst_npz, args.fc, args.bw)


if __name__ == "__main__":
    main()
