"""
Align reward and CST simulation counts on a dual-axis plot.

Data files (JSON) are read from ../code/data_rl/:
  tb_reward.json   — reward vs step
  tb_cst_sim.json  — CST simulation count vs step
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FIG_W = 7.2
FIG_H = 2.4
C_REWARD = "#1F6DB3"
C_CST = "#5A8F42"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 12, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "axes.linewidth": 1.0,
})


def parse(data):
    steps = np.array([row[1] for row in data], dtype=np.float64)
    values = np.array([row[2] for row in data], dtype=np.float64)
    _, idx = np.unique(steps, return_index=True)
    idx = np.sort(idx)
    return steps[idx], values[idx]


def main():
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "code", "data_rl")
    reward_raw = json.load(open(os.path.join(DATA_DIR, "tb_reward.json")))
    cst_raw = json.load(open(os.path.join(DATA_DIR, "tb_cst_sim.json")))

    r_step, r_val = parse(reward_raw)
    c_step_raw, c_val_raw = parse(cst_raw)

    # CST step is at episode level (every 25 RL steps); multiply by 25
    c_step = c_step_raw * 25
    c_val = c_val_raw

    # Align: union of steps, interpolate
    common_step = np.union1d(r_step.astype(int), c_step.astype(int)).astype(np.float64)
    r_interp = np.interp(common_step, r_step, r_val)
    c_interp = np.interp(common_step, c_step, c_val)

    c_step_events = c_step
    c_val_events = c_val

    # -- Plot ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))

    ax1.plot(common_step, r_interp, color=C_REWARD, lw=2.4, alpha=0.96,
             label="Reward")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward", color=C_REWARD)
    ax1.tick_params(axis="y", labelcolor=C_REWARD)
    ax1.set_xlim(common_step.min(), common_step.max())
    ax1.set_ylim(0.0, max(0.8, float(r_interp.max()) * 1.08))
    ax2 = ax1.twinx()
    ax2.plot(common_step, c_interp, color=C_CST, lw=2.3, alpha=0.92,
             label="CST simulations")
    ax2.set_ylabel("CST simulations", color=C_CST)
    ax2.tick_params(axis="y", labelcolor=C_CST)
    ax2.set_ylim(0.0, float(c_interp.max()) * 1.08)

    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.spines["left"].set_color(C_REWARD)
    ax2.spines["right"].set_color(C_CST)
    ax1.spines["bottom"].set_color("#444444")
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower right", frameon=False, fontsize=12)

    fig.tight_layout(pad=0.35)

    OUT_DIR = os.path.join(SCRIPT_DIR, "..", "code", "results_figures")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "rl_training_reward_cst.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Steps: {len(common_step)}, Reward range: [{r_val.min():.3f}, {r_val.max():.3f}]")
    print(f"CST calls: {int(c_val.max())} at step {int(c_step_events[-1])}")


if __name__ == "__main__":
    main()
