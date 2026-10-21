"""Tier-1 RL hyperparameter ablation launcher (mode agent, no CST).

Matrix: 5 configs x 3 seeds = 15 runs at 100k steps with the formal
agents' exact training conditions — the v11 trainer (terminal sparam
reward, fixed fc/bw grid, target-mask observations) and its rollout
geometry (episode-len 25 x selfplay-round-episodes 64 => PPO n_steps
1600). Runs sequentially; each run is tagged (ref / ent0 / ent0p1 /
clip0p1 / clip0p3) and seeded so aggregate_rl_hp_ablation.py can fold
them into a per-config mean +- SD table.

TensorBoard: the trainer already logs cycle/mean_reward, cycle/ and
train/global_best_reward per run under <run_dir>/tb/.

Usage:
  python run_rl_hp_ablation.py --dry-run          # print the 15 commands
  python run_rl_hp_ablation.py                    # run all pending
  python run_rl_hp_ablation.py --tags ref ent0    # subset
"""
import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(SCRIPT_DIR, "train_rl_active_learning_world_model.py")

DEFAULT_SEEDS = [42, 43, 44]
CONFIGS = [
    {"tag": "ref",     "ent_coef": 0.02, "clip_range": 0.2},
    {"tag": "ent0",    "ent_coef": 0.0,  "clip_range": 0.2},
    {"tag": "ent0p1",  "ent_coef": 0.1,  "clip_range": 0.2},
    {"tag": "clip0p1", "ent_coef": 0.02, "clip_range": 0.1},
    {"tag": "clip0p3", "ent_coef": 0.02, "clip_range": 0.3},
]
# Formal mode agent geometry (results_world_model/mode_transmit_*):
# ppo_n_steps = episode_len * selfplay_round_episodes = 25 * 64 = 1600,
# pretrained world model m1 = the same server checkpoint the formal
# agents used.
EPISODE_LEN = 25
SELFPLAY_ROUND_EPISODES = 64
DEFAULT_TIMESTEPS = 100000
WORLD_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "server",
    "best_system_eq_robust_dropout_M1_c_theta0p0to60p0_order16.pth")
DEFAULT_ROOT = os.path.join(SCRIPT_DIR, "results_world_model")


def build_matrix(configs, seeds):
    """Config-major order: every seed of one config before the next config."""
    return [{"tag": c["tag"], "ent_coef": c["ent_coef"],
             "clip_range": c["clip_range"], "seed": s}
            for c in configs for s in seeds]


def build_command(run, timesteps=DEFAULT_TIMESTEPS,
                  task_kind="mode", design_mode="transmit",
                  python=sys.executable):
    return [python, TRAINER,
            "--task-kind", task_kind,
            "--design-mode", design_mode,
            "--no-cst",
            "--timesteps", str(timesteps),
            "--episode-len", str(EPISODE_LEN),
            "--selfplay-round-episodes", str(SELFPLAY_ROUND_EPISODES),
            "--seed", str(run["seed"]),
            "--world-model", "m1",
            "--world-model-path", WORLD_MODEL_PATH,
            "--ent-coef", repr(float(run["ent_coef"])),
            "--clip-range", repr(float(run["clip_range"])),
            "--tag", run["tag"]]


def find_existing(root, tag, seed):
    """First run dir whose run_meta.json matches tag+seed, else None."""
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        meta_path = os.path.join(root, name, "run_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = json.load(open(meta_path))
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("tag") == tag and meta.get("seed") == seed:
            return os.path.join(root, name)
    return None


def partition_existing(root, matrix):
    """Split the matrix into (pending, skipped) by existing run_meta."""
    todo, skipped = [], []
    for run in matrix:
        done = find_existing(root, run["tag"], run["seed"])
        (skipped if done is not None else todo).append(run)
    return todo, skipped


def run_runs(commands, runs, jobs=1, cwd=SCRIPT_DIR, log_dir=None):
    """Execute (command, run) pairs, up to `jobs` in parallel.

    Returns [(run, returncode)] in submission order. With jobs > 1 each
    run's stdout/stderr is teed into log_dir/<tag>_s<seed>.log so the
    interleaved processes do not garble the console; with jobs == 1 the
    trainer's output streams straight through.
    """
    pairs = list(zip(commands, runs))
    if jobs <= 1:
        results = []
        for cmd, run in pairs:
            print(f"\n=== {run['tag']}/s{run['seed']} ===")
            rc = subprocess.run(cmd, cwd=cwd).returncode
            results.append((run, rc))
        return results

    import concurrent.futures

    if log_dir is None:
        log_dir = os.path.join(cwd, "ablation_logs")
    os.makedirs(log_dir, exist_ok=True)
    results = [None] * len(pairs)

    def _launch(item):
        idx, (cmd, run) = item
        log_path = os.path.join(log_dir, f"{run['tag']}_s{run['seed']}.log")
        print(f"[launch] {run['tag']}/s{run['seed']} -> {log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=cwd, stdout=log,
                                  stderr=subprocess.STDOUT)
        rc = proc.returncode
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[done]   {run['tag']}/s{run['seed']}: {status}", flush=True)
        return idx, run, rc

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for idx, run, rc in pool.map(_launch, enumerate(pairs)):
            results[idx] = (run, rc)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--tags", nargs="*", default=[c["tag"] for c in CONFIGS])
    ap.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel trainer processes (each writes its own "
                         "run dir; logs go to <root>/ablation_logs/).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if a matching run_meta.json exists")
    args = ap.parse_args(argv)

    configs = [c for c in CONFIGS if c["tag"] in set(args.tags)]
    matrix = build_matrix(configs, args.seeds)
    if args.force:
        todo, skipped = matrix, []
    else:
        todo, skipped = partition_existing(args.root, matrix)

    print(f"Matrix: {len(matrix)} runs ({len(configs)} configs x "
          f"{len(args.seeds)} seeds), timesteps={args.timesteps}")
    for run in skipped:
        print(f"  SKIP {run['tag']}/s{run['seed']} (already in {args.root})")
    for run in todo:
        print(f"  RUN  {run['tag']}/s{run['seed']}: "
              + " ".join(build_command(run, args.timesteps)[1:]))
    if args.dry_run:
        return 0

    commands = [build_command(run, args.timesteps) for run in todo]
    results = run_runs(commands, todo, jobs=args.jobs,
                       cwd=SCRIPT_DIR, log_dir=os.path.join(args.root,
                                                            "ablation_logs"))
    failures = [(run, rc) for run, rc in results if rc != 0]
    for run, rc in failures:
        print(f"[ablation] FAILED rc={rc}: {run['tag']}/s{run['seed']}",
              file=sys.stderr)
    print(f"\n[ablation] done: {len(todo) - len(failures)}/{len(todo)} runs "
          f"-> {args.root}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
