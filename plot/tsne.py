"""
RL design vs dataset analysis: reward distribution + t-SNE + novelty
=====================================================================
  1. Reward distribution histogram: RL vs Dataset
  2. t-SNE colored by reward (pattern space + S-param space)
  3. Distance to nearest dataset neighbour vs Reward (novelty)

Data is read from:
  - ../code/results_v11_world_model/<run>/  (RL round_*_best.npz + global_best.npz)
  - ../code/dataset_forward_mode_angle_v2.pt (dataset, via torch)
"""

import os
import random
import glob

import numpy as np

import torch  # only for loading the .pt dataset file, not from code/

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -- Single-column paper style -------------------------------------
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

C_S11 = "#1F6DB3"
C_S21 = "#C0182F"
C_RL = C_S11
C_DS = "#F0D58A"
C_BEST = C_S11
C_RL_EDGE = "#000000"
C_REWARD_MID = "#F8EDC7"
REWARD_CMAP = plt.get_cmap("coolwarm")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "code", "results_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths
RL_DIR = os.path.join(SCRIPT_DIR, "..", "code", "results_v11_world_model",
                       "mode_transmit_20260707_000423")
DATASET_PT = os.path.join(SCRIPT_DIR, "..", "code", "dataset_forward_mode_angle_v2.pt")

N_DATASET = 3000
RANDOM_SEED = 42
TARGET_FC = 10.0
TARGET_BW = 1.0

# -- Reward computation (pure numpy, no code/ deps) -----------------

V11_GUARD_GHZ = 0.25
V11_PASS_SCORE_TARGET = 0.89
V11_OUT_PENALTY_TARGET = 0.31


def _mag01(values):
    return np.clip(np.abs(np.asarray(values, dtype=np.float64)), 0.0, 1.0)


def _safe_complement(mag):
    mag = _mag01(mag)
    return np.sqrt(np.maximum(0.0, 1.0 - mag * mag))


def _band_mask(freqs, spec):
    freqs = np.asarray(freqs, dtype=np.float64)
    fc = float(spec["fc"])
    bw = float(spec["bw"])
    mask = (freqs >= fc - bw / 2.0) & (freqs <= fc + bw / 2.0)
    if not np.any(mask):
        mask = np.zeros_like(freqs, dtype=bool)
        mask[int(np.argmin(np.abs(freqs - fc)))] = True
    return mask


def _outband_mask(freqs, spec):
    freqs = np.asarray(freqs, dtype=np.float64)
    fc = float(spec["fc"])
    bw = float(spec["bw"])
    guard = max(0.0, float(spec.get("guard_ghz", V11_GUARD_GHZ)))
    lo = fc - bw / 2.0 - guard
    hi = fc + bw / 2.0 + guard
    return (freqs < lo) | (freqs > hi)


def _mean_or_zero(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if values.size else 0.0


def _transmission_score(s11, s21):
    s11 = _mag01(s11)
    s21 = _mag01(s21)
    return 0.5 * (_mean_or_zero(s21) + _mean_or_zero(_safe_complement(s11)))


def _reflection_score(s11, s21):
    s11 = _mag01(s11)
    s21 = _mag01(s21)
    return 0.5 * (_mean_or_zero(s11) + _mean_or_zero(_safe_complement(s21)))


def compute_reward(s11_mag, s21_mag, freqs_ghz, spec):
    """Compute v11 reward from magnitude arrays (numpy)."""
    s11 = _mag01(s11_mag)
    s21 = _mag01(s21_mag)
    freqs = np.asarray(freqs_ghz, dtype=np.float64)
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

    return float(in_score - out_penalty)


# -- Helper ---------------------------------------------------------

def interp_to_200(arr, x_src):
    x_dst = np.linspace(0.01, 20, 200)
    return np.interp(x_dst, x_src, np.asarray(arr, dtype=np.float64))


# -- RL data --------------------------------------------------------

def load_rl_data():
    """Load all RL round + global_best pattern / S21 / reward."""
    files = sorted(glob.glob(os.path.join(RL_DIR, "round_*_best.npz")))
    gb_path = os.path.join(RL_DIR, "global_best.npz")

    patterns, s21_feat, rewards = [], [], []

    for f in files + [gb_path]:
        d = np.load(f)
        pat = d["pattern_10x10"].flatten()
        patterns.append(pat)

        freqs = d["freqs_ghz"]
        x_src = np.linspace(0.01, 20, len(freqs))
        te = interp_to_200(d["pred_s21_pol0_theta0"], x_src)
        tm = interp_to_200(d["pred_s21_pol1_theta0"], x_src)
        s21_feat.append(np.concatenate([te, tm]))

        r_te = compute_reward(d["pred_s11_pol0_theta0"],
                               d["pred_s21_pol0_theta0"],
                               freqs, {"type": "transmit", "pol": 0,
                                       "theta": 0.0, "fc": TARGET_FC, "bw": TARGET_BW})
        r_tm = compute_reward(d["pred_s11_pol1_theta0"],
                               d["pred_s21_pol1_theta0"],
                               freqs, {"type": "reflect", "pol": 1,
                                       "theta": 0.0, "fc": TARGET_FC, "bw": TARGET_BW})
        rewards.append((r_te + r_tm) / 2)

    n_rnd = len(files)
    gb_pat, gb_sp, gb_r = patterns[-1], s21_feat[-1], rewards[-1]
    print(f"RL rounds: {n_rnd}, best reward: {gb_r:.3f}")
    return (np.array(patterns[:n_rnd]), np.array(s21_feat[:n_rnd]),
            np.array(rewards[:n_rnd]), gb_pat, gb_sp, gb_r)


# -- Dataset data (from .pt file) ---------------------------------

def load_dataset_data(n=N_DATASET, seed=RANDOM_SEED):
    """Sample N geometries from the .pt dataset and compute rewards."""
    print(f"Loading dataset from: {DATASET_PT}")
    data_obj = torch.load(DATASET_PT, weights_only=False)

    matrices = data_obj["matrices"].numpy()          # (N_total, 1, 10, 10)
    theta_deg = data_obj["theta_deg"].numpy()        # (N_total,)
    mode_id = data_obj["mode_id"].numpy()             # (N_total,)
    sample_id = data_obj["sample_id"].numpy()         # (N_total,)
    targets = data_obj["targets"].numpy()             # (N_total, 4, 2001)
    freqs_pt = data_obj["freqs"].numpy()              # (2001,)

    # Build pairs: TE (mode_id==0) and TM (mode_id==1) with same sample_id
    # Filter by theta range [0, 60] (matching original script behaviour)
    n_total = len(sample_id)
    te_rows = {}
    tm_rows = {}
    for i in range(n_total):
        sid = int(sample_id[i])
        theta = float(theta_deg[i])
        if theta < 0.0 or theta > 60.0:
            continue
        if mode_id[i] == 0:
            te_rows[sid] = i
        else:
            tm_rows[sid] = i

    paired = []
    for sid in te_rows:
        if sid in tm_rows:
            paired.append((te_rows[sid], tm_rows[sid]))

    print(f"Total paired TE/TM entries: {len(paired)}")

    rng = random.Random(seed)
    idxs = rng.sample(paired, min(n, len(paired)))

    patterns, s21_feat, rewards = [], [], []
    freq_ghz = freqs_pt  # already in GHz

    for te_idx, tm_idx in idxs:
        te_targ = targets[te_idx]   # (4, 2001)
        tm_targ = targets[tm_idx]   # (4, 2001)

        # targets: [s11_re, s11_im, s21_re, s21_im]
        te_s11 = np.sqrt(te_targ[0]**2 + te_targ[1]**2)
        te_s21 = np.sqrt(te_targ[2]**2 + te_targ[3]**2)
        tm_s11 = np.sqrt(tm_targ[0]**2 + tm_targ[1]**2)
        tm_s21 = np.sqrt(tm_targ[2]**2 + tm_targ[3]**2)

        theta_val = float(theta_deg[te_idx])

        pat = matrices[te_idx].flatten()
        patterns.append(pat)

        x_src = np.linspace(0.01, 20, 2001)
        te = interp_to_200(te_s21, x_src)
        tm = interp_to_200(tm_s21, x_src)
        s21_feat.append(np.concatenate([te, tm]))

        r_te = compute_reward(te_s11, te_s21, freq_ghz,
                              {"type": "transmit", "pol": 0, "theta": theta_val,
                               "fc": TARGET_FC, "bw": TARGET_BW})
        r_tm = compute_reward(tm_s11, tm_s21, freq_ghz,
                              {"type": "reflect", "pol": 1, "theta": theta_val,
                               "fc": TARGET_FC, "bw": TARGET_BW})
        rewards.append((r_te + r_tm) / 2)

    ds_pat = np.array(patterns)
    ds_sp = np.array(s21_feat)
    ds_reward = np.array(rewards)
    print(f"Dataset sampled: {len(ds_pat)}, reward max={ds_reward.max():.3f}")
    return ds_pat, ds_sp, ds_reward


# -- Plot helpers ---------------------------------------------------

def plot_reward_histogram(rl_r, ds_r, gb_r):
    """Fig 1: Reward distribution histogram."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    bins = np.linspace(-6, 2, 55)
    ax.hist(ds_r, bins=bins, color=C_DS, alpha=0.45, edgecolor="none",
            label=f"Dataset (n={len(ds_r)})")
    ax.hist(rl_r, bins=bins, color=C_RL, alpha=0.70, edgecolor="white",
            linewidth=0.3, label=f"RL (n={len(rl_r)})")

    ax.axvline(gb_r, color=C_BEST, lw=1.5, ls="--", alpha=0.85,
               label=f"Global Best ({gb_r:.2f})")
    ax.axvline(np.max(ds_r), color=C_DS, lw=1.0, ls=":", alpha=0.7,
               label=f"DS Max ({np.max(ds_r):.2f})")

    ax.set_xlabel("Reward (mean TE + TM)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              framealpha=0.45, edgecolor="#cccccc", fontsize=6,
              handlelength=1.2, handletextpad=0.5, borderpad=0.4)

    fig.tight_layout(pad=0.3)
    out = os.path.join(OUTPUT_DIR, "reward_hist_rl_vs_dataset.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_tsne_colored(emb, rewards, categories, gb_emb, gb_r,
                      out_name="tsne.png"):
    """Fig 2/3: t-SNE colored by reward."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    all_r = np.asarray(rewards)
    vmin = float(np.percentile(all_r, 2))
    vmax = float(np.percentile(all_r, 98))

    mask_ds = np.array(categories) == "dataset"
    mask_rl = np.array(categories) == "rl"

    sc_ds = ax.scatter(
        emb[mask_ds, 0], emb[mask_ds, 1],
        c=rewards[mask_ds], s=20, alpha=0.34,
        cmap=REWARD_CMAP, vmin=vmin, vmax=vmax,
        edgecolors="none", linewidths=0.0,
        marker="o", zorder=1,
    )

    sc_rl = ax.scatter(
        emb[mask_rl, 0], emb[mask_rl, 1],
        c=rewards[mask_rl], s=30, alpha=0.94,
        cmap=REWARD_CMAP, vmin=vmin, vmax=vmax,
        edgecolors=C_RL_EDGE, linewidths=1.0,
        marker="D", zorder=2,
    )

    ax.scatter(
        gb_emb[0, 0], gb_emb[0, 1],
        facecolors="white", s=100, marker="*",
        edgecolors=C_BEST, linewidths=1.0, zorder=3,
    )

    cbar = plt.colorbar(sc_rl, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Reward", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.8)

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DS,
               markeredgecolor="none", markersize=3.4, label="Dataset"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=REWARD_CMAP(0.82),
               markeredgecolor=C_RL_EDGE, markeredgewidth=0.8,
               markersize=3.5, label="RL Design"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor=C_BEST, markeredgewidth=1.0,
               markersize=5.6, label="Global Best"),
    ]
    ax.legend(handles=leg, loc="lower right", frameon=True, fancybox=False,
              framealpha=0.82, facecolor="white", edgecolor="#D8D8D8", fontsize=6,
              handlelength=1.0, handletextpad=0.40, borderpad=0.28)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(False)

    fig.tight_layout(pad=0.3)
    out = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_novelty(rl_pat, ds_pat, rl_r, gb_pat, gb_r):
    """Fig 4: Min Hamming distance to dataset vs Reward."""
    nn = NearestNeighbors(n_neighbors=1, metric="hamming")
    nn.fit(ds_pat)

    rl_dist, _ = nn.kneighbors(rl_pat, return_distance=True)
    rl_dist = rl_dist.flatten()
    gb_dist, _ = nn.kneighbors(gb_pat.reshape(1, -1), return_distance=True)

    # DS self-distance (sample 1000, exclude self)
    rng = random.Random(RANDOM_SEED)
    ds_idx = rng.sample(range(len(ds_pat)), min(1000, len(ds_pat)))
    ds_dist, _ = nn.kneighbors(ds_pat[ds_idx], return_distance=True)
    ds_dist = ds_dist.flatten()
    ds_dist = ds_dist[ds_dist > 0]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # DS self-distance rug
    ax.scatter(ds_dist, np.full_like(ds_dist, -0.03),
               c=C_DS, s=4, alpha=0.30, edgecolors="none",
               marker="|", linewidths=0.5)

    # RL
    ax.scatter(rl_dist, rl_r, c=C_RL, s=10, alpha=0.70,
               edgecolors="white", linewidths=0.3, zorder=2)

    # Global best
    ax.scatter(gb_dist[0], gb_r, c="white", s=38, marker="*",
               edgecolors=C_BEST, linewidths=0.8, zorder=3)

    ax.set_xlabel("Min Hamming distance to dataset")
    ax.set_ylabel("Reward")

    leg = [
        Line2D([0], [0], marker="|", color="w", markerfacecolor=C_DS,
               markersize=3.3, label="Dataset (rug)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_RL,
               markeredgecolor="white", markersize=4, label="RL Design"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor=C_BEST, markersize=8, label="Global Best"),
    ]
    ax.legend(handles=leg, loc="lower right", frameon=True, fancybox=False,
              framealpha=0.45, edgecolor="#cccccc", fontsize=6,
              handlelength=1.2, handletextpad=0.4, borderpad=0.35)

    fig.tight_layout(pad=0.3)
    out = os.path.join(OUTPUT_DIR, "novelty_rl_vs_dataset.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  RL mean dist to DS: {float(rl_dist.mean()):.4f}, "
          f"GB dist: {float(gb_dist[0]):.4f}")


# -- Main -----------------------------------------------------------

def main():
    print("Loading RL data...")
    rl_pat, rl_sp, rl_r, gb_pat, gb_sp, gb_r = load_rl_data()

    print("Loading dataset...")
    ds_pat, ds_sp, ds_r = load_dataset_data()

    # -- Fig 1: Reward histogram --
    print("\n=== Fig 1: Reward Histogram ===")
    plot_reward_histogram(rl_r, ds_r, gb_r)

    # -- Fig 2: Pattern space t-SNE --
    print("\n=== Fig 2: t-SNE Pattern Space ===")
    X_pat = np.vstack([rl_pat, ds_pat])
    emb_pat = TSNE(n_components=2, perplexity=40, random_state=RANDOM_SEED,
                    max_iter=1000).fit_transform(
                        np.vstack([X_pat, gb_pat.reshape(1, -1)]))
    all_r_pat = np.concatenate([rl_r, ds_r, [gb_r]])
    cats_pat = ["rl"] * len(rl_r) + ["dataset"] * len(ds_r) + ["rl"]
    plot_tsne_colored(emb_pat, all_r_pat, cats_pat,
                      emb_pat[-1:], gb_r,
                      out_name="tsne_pattern_reward.png")

    # -- Fig 3: S-param space t-SNE --
    print("\n=== Fig 3: t-SNE S-param Space ===")
    X_sp = np.vstack([rl_sp, ds_sp])
    emb_sp = TSNE(n_components=2, perplexity=40, random_state=RANDOM_SEED,
                   max_iter=1000).fit_transform(
                       np.vstack([X_sp, gb_sp.reshape(1, -1)]))
    plot_tsne_colored(emb_sp, all_r_pat, cats_pat,
                      emb_sp[-1:], gb_r,
                      out_name="tsne_sparam_reward.png")

    # -- Fig 4: Novelty --
    print("\n=== Fig 4: Novelty ===")
    plot_novelty(rl_pat, ds_pat, rl_r, gb_pat, gb_r)


if __name__ == "__main__":
    main()
