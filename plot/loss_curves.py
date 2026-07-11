"""
Four-model loss comparison for 0-60 degree forward-model ablation.

Data priority:
  1. loss_history.csv in each result directory, if present
  2. legacy training log in logs/

Reads data from ../code/code0701/code0701_server/
Output to ../code/results_figures/
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(SCRIPT_DIR, "..", "code", "code0701", "code0701_server")
LOG_DIR = os.path.join(SERVER_DIR, "logs")
RESULT_DIR = os.path.join(SERVER_DIR, "results")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "code", "results_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_TRAIN_BIN = 1
VAL_STRIDE = 10
MAX_EPOCH = 300

DEFAULT_FIG_W = 3.5
DEFAULT_FIG_H = 2.65

MODEL_CONFIG = {
    "CNN": {
        "log": "cnn_theta_pol_0to60.log",
        "result": "cnn_theta_pol_taylor_theta0p0to60p0_order16",
        "color": "#4C78A8",
    },
    "DeepONet": {
        "log": "deeponet_theta_pol_0to60.log",
        "result": "deeponet_theta_pol_taylor_theta0p0to60p0_order16",
        "color": "#F58518",
    },
    "SE-DeepONet": {
        "log": "se_deeponet_theta_pol_0to60.log",
        "result": "se_deeponet_theta_pol_taylor_theta0p0to60p0_order16",
        "color": "#54A24B",
    },
    "MIONet": {
        "log": "m1_cheb_theta0to60.log",
        "result": "m1_chebyshev_theta0p0to60p0_order16",
        "color": "#E45756",
    },
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.8,
    "axes.linewidth": 1.05,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

LOSS_RE = re.compile(r"(\d+)/(\d+):.*?loss=([0-9eE.+-]+)")
VAL_RE = re.compile(r"epoch=(\d+)/(\d+)\s+train=([0-9eE.+-]+)\s+val=([0-9eE.+-]+)")


def _as_arrays(points):
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs, ys = zip(*points)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _bin_train(epoch_losses, max_epoch=MAX_EPOCH, train_bin=DEFAULT_TRAIN_BIN):
    points = []
    train_bin = max(1, int(train_bin))
    for start in range(1, max_epoch + 1, train_bin):
        end = min(start + train_bin - 1, max_epoch)
        values = []
        for epoch in range(start, end + 1):
            values.extend(epoch_losses.get(epoch, []))
        if values:
            x = 0.5 * (start + end)
            points.append((x, float(np.mean(values))))
    return _as_arrays(points)


def _sample_val(epoch_values, max_epoch=MAX_EPOCH, val_stride=VAL_STRIDE):
    points = []
    val_stride = max(1, int(val_stride))
    for epoch in sorted(epoch_values):
        if epoch == 1 or epoch % val_stride == 0 or epoch == max_epoch:
            points.append((epoch, epoch_values[epoch]))
    return _as_arrays(points)


def load_loss_history(result_dir, train_bin=DEFAULT_TRAIN_BIN, val_stride=VAL_STRIDE):
    csv_path = os.path.join(result_dir, "loss_history.csv")
    if not os.path.exists(csv_path):
        return None

    epoch_train = defaultdict(list)
    epoch_val = {}
    max_epoch = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            max_epoch = max(max_epoch, epoch)
            epoch_train[epoch].append(float(row["train_loss"]))
            val_text = str(row.get("val_loss", "")).strip()
            if val_text:
                epoch_val[epoch] = float(val_text)

    train_x, train_y = _bin_train(epoch_train, max_epoch=max_epoch or MAX_EPOCH,
                                  train_bin=train_bin)
    val_x, val_y = _sample_val(epoch_val, max_epoch=max_epoch or MAX_EPOCH,
                               val_stride=val_stride)
    return train_x, train_y, val_x, val_y


def parse_log(log_path, train_bin=DEFAULT_TRAIN_BIN, val_stride=VAL_STRIDE):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    epoch_losses = defaultdict(list)
    max_epoch = MAX_EPOCH
    for match in LOSS_RE.finditer(text):
        epoch = int(match.group(1))
        max_epoch = max(max_epoch, int(match.group(2)))
        epoch_losses[epoch].append(float(match.group(3)))

    epoch_val = {}
    for match in VAL_RE.finditer(text):
        epoch = int(match.group(1))
        max_epoch = max(max_epoch, int(match.group(2)))
        epoch_val[epoch] = float(match.group(4))

    train_x, train_y = _bin_train(epoch_losses, max_epoch=max_epoch, train_bin=train_bin)
    val_x, val_y = _sample_val(epoch_val, max_epoch=max_epoch, val_stride=val_stride)
    return train_x, train_y, val_x, val_y


def load_model_curve(cfg, train_bin=DEFAULT_TRAIN_BIN, val_stride=VAL_STRIDE):
    result_dir = os.path.join(RESULT_DIR, cfg["result"])
    history = load_loss_history(result_dir, train_bin=train_bin, val_stride=val_stride)
    if history is not None and len(history[0]) > 0:
        return (*history, "loss_history.csv")

    log_path = os.path.join(LOG_DIR, cfg["log"])
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)
    return (*parse_log(log_path, train_bin=train_bin, val_stride=val_stride), "legacy log")


def style_axes(ax):
    ax.set_yscale("log")
    ax.set_xlim(0, MAX_EPOCH + 5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, which="major", color="#D9D9D9", linestyle=":", linewidth=0.65, alpha=0.78)
    ax.grid(True, which="minor", color="#EDEDED", linestyle=":", linewidth=0.38, alpha=0.42)
    ax.tick_params(which="major", length=4.2, width=1.05, top=False, right=False, pad=2.0)
    ax.tick_params(which="minor", length=2.6, width=0.8, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot four-model train/validation loss curves.")
    parser.add_argument("--train-bin", type=int, default=DEFAULT_TRAIN_BIN,
                        help="Epoch bin size for train loss aggregation. Default: 1")
    parser.add_argument("--val-stride", type=int, default=VAL_STRIDE,
                        help="Epoch stride for validation loss display. Default: 10")
    parser.add_argument("--fig-w", type=float, default=DEFAULT_FIG_W,
                        help="Figure width in inches. Default is single-column width.")
    parser.add_argument("--fig-h", type=float, default=DEFAULT_FIG_H,
                        help="Figure height in inches. Default is single-column height.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))

    model_handles = []
    all_y = []
    summary = []

    for name, cfg in MODEL_CONFIG.items():
        try:
            train_ep, train_loss, val_ep, val_loss, source = load_model_curve(
                cfg, train_bin=args.train_bin, val_stride=args.val_stride)
        except FileNotFoundError as exc:
            print(f"SKIP {name}: missing {exc}")
            continue

        color = cfg["color"]
        if len(train_loss):
            ax.plot(train_ep, train_loss, color=color, lw=1.28, ls="-", alpha=0.95,
                    solid_capstyle="round")
            all_y.extend(train_loss.tolist())
        if len(val_loss):
            ax.plot(val_ep, val_loss, color=color, lw=1.18, ls=(0, (5, 3)), alpha=0.30,
                    dash_capstyle="round")
            all_y.extend(val_loss.tolist())

        model_handles.append(Line2D([0], [0], color=color, lw=1.6, ls="-", label=name))
        final_train = float(train_loss[-1]) if len(train_loss) else float("nan")
        final_val = float(val_loss[-1]) if len(val_loss) else float("nan")
        summary.append((name, len(train_ep), len(val_ep), final_train, final_val, source))

    style_axes(ax)
    if all_y:
        ymin = max(min(all_y) * 0.72, 1e-5)
        ymax = max(all_y) * 1.45
        ax.set_ylim(ymin, ymax)

    train_handle = Line2D([0], [0], color="#222222", lw=1.7, ls="-", label="Train")
    val_handle = Line2D([0], [0], color="#222222", lw=1.5, ls=(0, (5, 3)), alpha=0.45, label="Val")
    handles = model_handles + [train_handle, val_handle]
    legend = ax.legend(handles=handles, loc="upper right", ncol=2, frameon=True,
                       handlelength=1.55, columnspacing=0.85, handletextpad=0.42,
                       borderaxespad=0.22, labelspacing=0.22, fancybox=False,
                       framealpha=0.92, edgecolor="black", facecolor="white")
    legend.get_frame().set_linewidth(0.65)

    fig.tight_layout(pad=0.22)
    out_png = os.path.join(OUTPUT_DIR, "loss_curves_4models.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)

    for name, n_train, n_val, final_train, final_val, source in summary:
        print(f"{name:12s} source={source:16s} train_pts={n_train:3d} val_pts={n_val:2d} "
              f"final_train={final_train:.6g} final_val={final_val:.6g}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
