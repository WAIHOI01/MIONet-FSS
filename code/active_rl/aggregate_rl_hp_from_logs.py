"""Aggregate RL HP ablation results straight from the trainer stdout logs.

Works when only <root>/ablation_logs/<tag>_s<seed>.log files are available
(no run dirs): parses the per-cycle lines

  [cycle N] ... mean_reward=X ... global_best=Y ...

and reproduces the same endpoints as aggregate_rl_hp_ablation.py:
  best_reward      = max global_best over cycles
  best_cycle_mean  = max cycle mean_reward
  mean_all         = mean of all cycle mean_rewards
  mean_last10      = mean of the final 10 cycle mean_rewards
Per-config mean ± SD (ddof=1) over seeds.

Usage:
  python aggregate_rl_hp_from_logs.py                       # console table
  python aggregate_rl_hp_from_logs.py --out hp_ablation.md  # + markdown
"""
import argparse
import glob
import os
import re

import numpy as np

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results_world_model", "ablation_logs")
CYCLE_RE = re.compile(
    r"\[cycle (\d+)\].*?mean_reward=(-?[\d.]+(?:[eE][+-]?\d+)?)"
    r".*?global_best=(-?[\d.]+(?:[eE][+-]?\d+)?)")
NAME_RE = re.compile(r"^(.+)_s(\d+)\.log$")
LAST_K = 10


def parse_log(path):
    """[(cycle, mean_reward, global_best)] from one trainer stdout log."""
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CYCLE_RE.search(line)
            if m:
                recs.append((int(m.group(1)), float(m.group(2)),
                             float(m.group(3))))
    return recs


def endpoints(recs):
    means = [r[1] for r in recs]
    best = max((r[2] for r in recs), default=None)
    last = means[-LAST_K:]
    return {
        "best_reward": best,
        "best_cycle_mean": max(means) if means else None,
        "mean_all": float(np.mean(means)) if means else None,
        "mean_last10": float(np.mean(last)) if last else None,
        "n_cycles": len(recs),
    }


def collect(log_dir):
    """[{tag, seed, **endpoints}] sorted by (tag, seed)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        recs = parse_log(path)
        row = {"tag": m.group(1), "seed": int(m.group(2)),
               "log": os.path.basename(path)}
        row.update(endpoints(recs))
        rows.append(row)
    rows.sort(key=lambda r: (r["tag"], r["seed"]))
    return rows


METRICS = ["best_reward", "best_cycle_mean", "mean_all", "mean_last10"]


def group_stats(rows):
    stats = {}
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    for tag, group in by_tag.items():
        stats[tag] = {}
        for metric in METRICS:
            vals = [r[metric] for r in group if r[metric] is not None]
            entry = {"n": len(vals), "mean": None, "sd": None}
            if vals:
                entry["mean"] = float(np.mean(vals))
                if len(vals) >= 2:
                    entry["sd"] = float(np.std(vals, ddof=1))
            stats[tag][metric] = entry
    return stats


def _fmt(e):
    if e["mean"] is None:
        return "-"
    if e["sd"] is None:
        return f"{e['mean']:.4f}"
    return f"{e['mean']:.4f} ± {e['sd']:.4f}"


def write_markdown(stats, rows, out_md):
    parent = os.path.dirname(os.path.abspath(out_md))
    os.makedirs(parent, exist_ok=True)
    lines = ["# RL hyperparameter ablation (mode agent, no CST, 100k steps)",
             "",
             "Per-config mean ± SD over seeds (sample SD, ddof=1); "
             "endpoints parsed from trainer logs.", "",
             "| config | n | best reward | best cycle-mean | mean all "
             "| mean last-10 |", "|---|---:|---|---|---|---|"]
    for tag in sorted(stats):
        s = stats[tag]
        lines.append(f"| {tag} | {s['best_reward']['n']} "
                     f"| {_fmt(s['best_reward'])} "
                     f"| {_fmt(s['best_cycle_mean'])} "
                     f"| {_fmt(s['mean_all'])} "
                     f"| {_fmt(s['mean_last10'])} |")
    lines += ["", "## Per-run endpoints", "",
              "| config | seed | best reward | best cycle-mean | mean all "
              "| mean last-10 | cycles |", "|---|---:|---|---|---|---|---:|"]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['seed']} | {r['best_reward']:.4f} "
            f"| {r['best_cycle_mean']:.4f} | {r['mean_all']:.4f} "
            f"| {r['mean_last10']:.4f} | {r['n_cycles']} |")
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = collect(args.log_dir)
    if not rows:
        print(f"No *.log files under {args.log_dir}")
        return 1
    stats = group_stats(rows)

    print(f"Per-config (mean ± SD over seeds), logs={args.log_dir}\n")
    print(f"{'config':10s} {'n':>2s}  {'best reward':18s} "
          f"{'best cycle-mean':18s} {'mean all':18s} {'mean last-10'}")
    for tag in sorted(stats):
        s = stats[tag]
        print(f"{tag:10s} {s['best_reward']['n']:2d}  "
              f"{_fmt(s['best_reward']):18s} {_fmt(s['best_cycle_mean']):18s} "
              f"{_fmt(s['mean_all']):18s} {_fmt(s['mean_last10'])}")
    print("\nPer-run:")
    for r in rows:
        print(f"  {r['tag']:10s} seed={r['seed']} best={r['best_reward']:.4f} "
              f"best_cycle_mean={r['best_cycle_mean']:.4f} "
              f"mean_all={r['mean_all']:.4f} "
              f"mean_last10={r['mean_last10']:.4f} "
              f"n_cycles={r['n_cycles']} ({r['log']})")
    if args.out:
        write_markdown(stats, rows, args.out)
        print(f"\n[aggregate] markdown -> {args.out}")
    return 0


if __name__ == "__main__":
    main()
