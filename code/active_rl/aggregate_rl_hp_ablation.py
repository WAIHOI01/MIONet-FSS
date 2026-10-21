"""Aggregate RL hyperparameter ablation runs into a per-config table.

Walks an ablation root (results_v11_world_model), finds run dirs with a
run_meta.json carrying {tag, seed, global_best_reward,
best_cycle_mean_reward} plus a cycle_metrics.jsonl (one {cycle,
mean_reward} per active-learning cycle), and folds them into per-config
mean +- SD over seeds (sample SD, ddof=1). Metrics:

  best_reward      global best single-design reward over the whole run
  best_cycle_mean  best per-cycle mean reward
  mean_all         mean cycle-mean reward over all cycles
  mean_last10      mean cycle-mean reward over the final 10 cycles
                   (fewer if the run has fewer cycles)

Usage:
  python aggregate_rl_hp_ablation.py                       # print table
  python aggregate_rl_hp_ablation.py --out hp_ablation.md  # markdown too
"""
import argparse
import json
import os

import numpy as np

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_v11_world_model")
LAST_K = 10


def collect(root):
    """Per-run endpoint rows: [{tag, seed, best_reward, best_cycle_mean,
    mean_all, mean_last10, n_cycles, dir}] sorted by (tag, seed)."""
    rows = []
    if not os.path.isdir(root):
        return rows
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        meta_path = os.path.join(run_dir, "run_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "tag" not in meta:
            continue
        cycle_means = []
        jsonl = os.path.join(run_dir, "cycle_metrics.jsonl")
        if os.path.exists(jsonl):
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "mean_reward" in rec:
                        cycle_means.append(float(rec["mean_reward"]))
        last = cycle_means[-LAST_K:] if LAST_K else cycle_means
        rows.append({
            "tag": meta["tag"],
            "seed": meta.get("seed"),
            "best_reward": meta.get("global_best_reward"),
            "best_cycle_mean": meta.get("best_cycle_mean_reward"),
            "mean_all": float(np.mean(cycle_means)) if cycle_means else None,
            "mean_last10": float(np.mean(last)) if last else None,
            "n_cycles": len(cycle_means),
            "dir": name,
        })
    rows.sort(key=lambda r: (str(r["tag"]),
                             r["seed"] if r["seed"] is not None else -1))
    return rows


METRICS = ["best_reward", "best_cycle_mean", "mean_all", "mean_last10"]


def group_stats(rows):
    """{tag: {metric: {"mean", "sd", "n"}}} — sample SD (ddof=1) over
    seeds; sd is None when the config has fewer than two runs."""
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


def _fmt(entry):
    if entry["mean"] is None:
        return "-"
    if entry["sd"] is None:
        return f"{entry['mean']:.4f}"
    return f"{entry['mean']:.4f} ± {entry['sd']:.4f}"


def write_markdown(stats, rows, out_md):
    lines = ["# RL hyperparameter ablation (mode agent, no CST)", "",
             "Per-config mean ± SD over seeds (sample SD, ddof=1).", "",
             "| config | n | best reward | best cycle-mean | mean all | mean last-10 |",
             "|---|---:|---|---|---|---|"]
    for tag in sorted(stats):
        s = stats[tag]
        lines.append(f"| {tag} | {s['best_reward']['n']} "
                     f"| {_fmt(s['best_reward'])} "
                     f"| {_fmt(s['best_cycle_mean'])} "
                     f"| {_fmt(s['mean_all'])} "
                     f"| {_fmt(s['mean_last10'])} |")
    lines += ["", "## Per-run endpoints", "",
              "| config | seed | best reward | best cycle-mean | mean all "
              "| mean last-10 | cycles |",
              "|---|---:|---|---|---|---|---:|"]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['seed']} "
            f"| {r['best_reward']:.4f} | {r['best_cycle_mean']:.4f} "
            f"| {r['mean_all']:.4f} | {r['mean_last10']:.4f} "
            f"| {r['n_cycles']} |")
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=None,
                    help="optional markdown output path")
    args = ap.parse_args(argv)

    rows = collect(args.root)
    stats = group_stats(rows)
    if not rows:
        print(f"No ablation runs (run_meta.json with tag) under {args.root}")
        return 0

    print(f"Per-config (mean ± SD over seeds), root={args.root}\n")
    print(f"{'config':10s} {'n':>2s}  {'best reward':16s} "
          f"{'best cycle-mean':16s} {'mean all':16s} {'mean last-10'}")
    for tag in sorted(stats):
        s = stats[tag]
        print(f"{tag:10s} {s['best_reward']['n']:2d}  "
              f"{_fmt(s['best_reward']):16s} {_fmt(s['best_cycle_mean']):16s} "
              f"{_fmt(s['mean_all']):16s} {_fmt(s['mean_last10'])}")
    print("\nPer-run:")
    for r in rows:
        print(f"  {r['tag']:10s} seed={r['seed']} best={r['best_reward']:.4f} "
              f"best_cycle_mean={r['best_cycle_mean']:.4f} "
              f"mean_all={r['mean_all']:.4f} mean_last10={r['mean_last10']:.4f} "
              f"n_cycles={r['n_cycles']} ({r['dir']})")
    if args.out:
        write_markdown(stats, rows, args.out)
        print(f"\n[aggregate] markdown -> {args.out}")
    return 0


if __name__ == "__main__":
    main()
