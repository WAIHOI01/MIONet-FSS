"""Command line interface: argument parsing and the train entry point.

The 27 flags of the former v9 parser keep their order, defaults and help
text; the 3 fixed-target flags of the former v11 pre-parser are now native
arguments appended at the end (same defaults).
"""

import argparse

from active_rl.config import (
    DEFAULT_STABILITY_WEIGHT,
    DEFAULT_CST_BUDGET,
    DEFAULT_CST_INTERVAL_EPISODES,
    DEFAULT_MIN_SURROGATE_REWARD,
    DEFAULT_MIN_ACTIVE_QUARTER_PIXELS,
    DEFAULT_INVALID_PATTERN_REWARD,
    UNCERTAINTY_THRESHOLD,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=50000)
    p.add_argument("--episode-len", type=int, default=25)
    p.add_argument("--selfplay-round-episodes", type=int, default=50,
                   help="Episodes per self-play round; PPO n_steps is episode_len * this value.")
    p.add_argument("--task-kind", choices=["joint", "angle", "mode"], default="joint")
    p.add_argument("--design-mode", choices=["transmit", "reflect", "both"], default="both")
    p.add_argument("--angle-pol", choices=["te", "tm"], default="te")
    p.add_argument("--stability-weight", type=float, default=DEFAULT_STABILITY_WEIGHT)
    p.add_argument("--world-model", choices=["m1", "m2"], default="m1")
    p.add_argument("--world-model-path", default=None)
    p.add_argument("--mock-cst", action="store_true")
    p.add_argument("--no-cst", action="store_true")
    p.add_argument("--cst-budget", type=int, default=DEFAULT_CST_BUDGET,
                   help="Maximum number of CST design queries; <=0 means unlimited.")
    p.add_argument("--cst-interval-episodes", type=int, default=DEFAULT_CST_INTERVAL_EPISODES)
    p.add_argument("--min-surrogate-reward", type=float, default=DEFAULT_MIN_SURROGATE_REWARD)
    p.add_argument("--min-active-quarter-pixels", type=int, default=DEFAULT_MIN_ACTIVE_QUARTER_PIXELS,
                   help="Minimum active pixels in the 5x5 quarter pattern; 1 prevents all-empty designs.")
    p.add_argument("--invalid-pattern-reward", type=float, default=DEFAULT_INVALID_PATTERN_REWARD,
                   help="Terminal reward assigned to designs violating simple pattern validity constraints.")
    p.add_argument("--uncertainty-threshold", type=float, default=UNCERTAINTY_THRESHOLD)
    p.add_argument("--cst-query-mode", choices=["representative", "full"], default="representative")
    p.add_argument("--cst-host", default="127.0.0.1")
    p.add_argument("--cst-port", type=int, default=65433)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sweep-top-n", type=int, default=4)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for numpy/python/torch/PPO; default keeps prior (unseeded) behavior.")
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="PPO learning rate (default matches the formal agents).")
    p.add_argument("--clip-range", type=float, default=0.2,
                   help="PPO clip range (default matches the formal agents).")
    p.add_argument("--ent-coef", type=float, default=0.02,
                   help="PPO entropy coefficient (default matches the formal agents).")
    p.add_argument("--tag", default=None,
                   help="Optional run tag (e.g. ablation config) appended to the run dir label and stored in run_meta.")
    p.add_argument("--target-fc-noise", type=float, default=0.0,
                   help="Uniform fc perturbation in GHz around the fixed v11 grid.")
    p.add_argument("--target-bw-noise", type=float, default=0.0,
                   help="Uniform bw perturbation in GHz around the fixed v11 grid.")
    p.add_argument("--target-sampling", choices=("grid", "continuous"),
                   default="grid",
                   help="Task sampling: 'grid' = fixed v11 4x4 (fc, bw) grid; "
                        "'continuous' = v9 uniform over the feasible ranges "
                        "(task distribution of paper run 183059).")
    return p.parse_args()


def main():
    args = parse_args()
    # Lazy import: argparse/--help must not pay the torch import cost.
    from active_rl.trainer import train
    train(args)


if __name__ == "__main__":
    main()
