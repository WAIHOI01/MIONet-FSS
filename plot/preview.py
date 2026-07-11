"""
Generate a preview figure from a global_best.npz file: pixel-pattern inset + TE/TM S-param curves.

Usage:
  python preview.py <npz_path> --out preview.png [--fc 10 --bw 1]

Reads a npz with keys: pattern_10x10, freqs_ghz,
  pred_s11_pol0_theta0, pred_s21_pol0_theta0,
  pred_s11_pol1_theta0, pred_s21_pol1_theta0
"""

import os
import sys
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_preview(pattern_10x10, pred_items, freqs_ghz, fc, bw,
                 model_key, task_kind, reward_val, out_png):
    """Generate preview: pixel-pattern + TE/TM S-param prediction curves.

    Parameters
    ----------
    pattern_10x10 : np.ndarray, shape (10, 10)
        Binary pixel pattern.
    pred_items : list of dict
        Each dict has keys: pol (0/1), s11 (ndarray), s21 (ndarray).
    freqs_ghz : np.ndarray
        Frequency vector.
    fc, bw : float
        Target centre frequency and bandwidth.
    model_key : str
        Model identifier for the title.
    task_kind : str
        Task type for the title.
    reward_val : float
        Reward value shown in the pattern subplot title.
    out_png : str
        Output PNG path.
    """
    C_S11 = "#1F6DB3"
    C_S21 = "#C0182F"
    EPS_M = 1e-8
    band = (fc - bw / 2, fc + bw / 2)
    FIG_W = 3.5
    FIG_H = 2.65

    te_item = next((p for p in pred_items if p["pol"] == 0), None)
    tm_item = next((p for p in pred_items if p["pol"] == 1), None)
    n_s_param_plots = 2 if tm_item is not None else 1
    n_total = n_s_param_plots + 1  # + pattern inset

    fig = plt.figure(figsize=(FIG_W * n_total, FIG_H))

    for col, (pol_name, item) in enumerate(
        [("TE", te_item), ("TM", tm_item)]):
        if item is None:
            continue
        ax = fig.add_subplot(1, n_total, col + 1)
        ax.plot(freqs_ghz, 20 * np.log10(np.abs(item["s11"]) + EPS_M),
                color=C_S11, lw=1.1, ls="-", alpha=0.95,
                label=r"$S_{11}$ Pred")
        ax.plot(freqs_ghz, 20 * np.log10(np.abs(item["s21"]) + EPS_M),
                color=C_S21, lw=1.1, ls="-", alpha=0.95,
                label=r"$S_{21}$ Pred")
        ax.axvspan(*band, alpha=0.10, color="#4CAF50")
        for b in band:
            ax.axvline(b, color="#4CAF50", lw=0.6, ls=":", alpha=0.6)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(r"$S$-parameter (dB)")
        ax.set_ylim(-50, 5)
        ax.set_xlim(0, 20)
        ax.set_title(f"{pol_name} Pred", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=6, framealpha=0.5)

    # Pixel pattern
    ax_pat = fig.add_subplot(1, n_total, n_total)
    ax_pat.imshow(pattern_10x10.T, cmap="binary", origin="lower", vmin=0, vmax=1)
    n_metal = int(pattern_10x10.sum())
    ax_pat.set_title(f"{n_metal}/100 metal\nr={reward_val:.3f}", fontsize=8)
    ax_pat.set_xticks([])
    ax_pat.set_yticks([])

    fig.suptitle(f"{model_key} | fc={fc:.0f} bw={bw:.0f} | {task_kind}",
                 fontsize=10, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate preview figure from global_best.npz")
    parser.add_argument("npz_path", help="Path to global_best.npz")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: same dir as npz / global_best.png)")
    parser.add_argument("--fc", type=float, default=10.0,
                        help="Target centre frequency (GHz)")
    parser.add_argument("--bw", type=float, default=1.0,
                        help="Target bandwidth (GHz)")
    parser.add_argument("--model-key", default="RL Design",
                        help="Model key for figure title")
    parser.add_argument("--task-kind", default="mode",
                        help="Task kind for figure title")
    args = parser.parse_args()

    d = np.load(args.npz_path)

    pattern = d["pattern_10x10"]
    freqs_ghz = d["freqs_ghz"]

    pred_items = []
    for pol in (0, 1):
        s11_key = f"pred_s11_pol{pol}_theta0"
        s21_key = f"pred_s21_pol{pol}_theta0"
        if s11_key in d and s21_key in d:
            pred_items.append({
                "pol": pol,
                "s11": d[s11_key],
                "s21": d[s21_key],
            })

    if not pred_items:
        print("ERROR: No pred_s11/pred_s21 keys found in the NPZ file.", file=sys.stderr)
        sys.exit(1)

    # Try to determine reward from the file name or compute a simple proxy
    reward_val = 0.0

    out_png = args.out
    if out_png is None:
        out_dir = os.path.dirname(args.npz_path)
        out_png = os.path.join(out_dir, "global_best.png")

    plot_preview(pattern, pred_items, freqs_ghz, args.fc, args.bw,
                 args.model_key, args.task_kind, reward_val, out_png)


if __name__ == "__main__":
    main()
