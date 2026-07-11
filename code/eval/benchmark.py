"""Unified benchmark: RL vs GA vs Random vs DS, configurable evaluation budget."""
import json
import os
import time

import numpy as np
import torch

from common.config import DEVICE, N_FREQ, STANDARD_FREQS
from forward.world_model import LocalEnsembleWorldModel
from inverse.rl_search import load_rl_model, rl_search, quarter_to_full
from inverse.ga_search import run_ga, run_random
from inverse.dataset_search import search_dataset
from inverse.task import build_active_design_task


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FC_LIST = [10.0, 12.0, 14.0]
DEFAULT_BW_LIST = [1.0, 2.0, 3.0, 4.0]
DEFAULT_TASK_KINDS = ["mode", "angle", "joint"]

# GA parameters
GA_POP = 20
GA_TOURNAMENT = 3
GA_CROSSOVER = 0.8
GA_MUTATION = 0.06
GA_ELITISM = 2
GA_EARLY_STOP = 8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_npz_results(out_dir, pattern, specs, wm, pred_items=None):
    """Save pattern and predicted S-parameters to NPZ.

    If *pred_items* is provided (RL case), those are used directly.
    Otherwise, WM prediction is run on *pattern* against *specs*.
    """
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "best.npz")

    freqs = np.linspace(0.01, 20.0, N_FREQ, dtype=np.float32)
    npz_d = {"pattern_10x10": pattern.astype(np.uint8), "freqs_ghz": freqs}

    if pred_items is not None:
        for pi in pred_items:
            pol = int(pi["pol"])
            th = int(pi["theta"])
            npz_d[f"pred_s11_pol{pol}_theta{th}"] = (
                pi["s11"].astype(np.float32)
            )
            npz_d[f"pred_s21_pol{pol}_theta{th}"] = (
                pi["s21"].astype(np.float32)
            )
    else:
        # Batch predict from world model
        ft = (
            torch.from_numpy(pattern).float().to(DEVICE)
            .unsqueeze(0).unsqueeze(0)
        )
        n_s = len(specs)
        Mt_b = ft.repeat(n_s, 1, 1, 1)
        thetas = torch.tensor(
            [float(s["theta"]) for s in specs], device=DEVICE,
        )
        pols = torch.tensor(
            [int(s["pol"]) for s in specs], device=DEVICE, dtype=torch.long,
        )
        with torch.no_grad():
            s11_b, s21_b, _ = wm.predict_stats(
                Mt_b, theta=thetas, pol=pols,
            )
        for i, s in enumerate(specs):
            pol = int(s["pol"])
            th = int(s["theta"])
            npz_d[f"pred_s11_pol{pol}_theta{th}"] = (
                s11_b[i].detach().cpu().numpy().flatten().astype(np.float32)
            )
            npz_d[f"pred_s21_pol{pol}_theta{th}"] = (
                s21_b[i].detach().cpu().numpy().flatten().astype(np.float32)
            )

    np.savez(npz_path, **npz_d)
    return npz_path


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def run_benchmark(model_dirs, fc_list=None, bw_list=None, task_kinds=None,
                  n_evals=200, output_dir=None, world_model_path=None,
                  verbose=True):
    """Run complete benchmark comparing RL, GA, Random, and Dataset oracle.

    Parameters
    ----------
    model_dirs : dict
        Maps task kind (``"mode"``, ``"angle"``, ``"joint"``) to the RL
        model directory path (containing the PPO checkpoint zip).
    fc_list : list of float, optional
        Centre frequencies.  Defaults to ``[10.0, 12.0, 14.0]``.
    bw_list : list of float, optional
        Bandwidths.  Defaults to ``[1.0, 2.0, 3.0, 4.0]``.
    task_kinds : list of str, optional
        Task kinds.  Defaults to ``["mode", "angle", "joint"]``.
    n_evals : int
        Evaluation (query) budget per method per task.  Default 200.
    output_dir : str, optional
        Root directory for results.  If None, a timestamped directory is
        created under the current working directory.
    world_model_path : str, optional
        Path to the MIONet checkpoint.  Defaults to the configured path.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Keys:

        - ``"summary"`` : dict  -- method-level aggregate statistics
        - ``"results"`` : dict  -- per-method list of per-task result dicts
        - ``"output_dir"`` : str
    """
    # Defaults
    if fc_list is None:
        fc_list = DEFAULT_FC_LIST
    if bw_list is None:
        bw_list = DEFAULT_BW_LIST
    if task_kinds is None:
        task_kinds = DEFAULT_TASK_KINDS

    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(),
            f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Unified Benchmark: RL vs GA vs Random vs DS")
        print(f"  Tasks: {len(fc_list) * len(bw_list)} per kind "
              f"x {len(task_kinds)} kinds")
        print(f"  Evaluations per task: {n_evals}")
        print(f"  Output: {output_dir}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Load shared world model
    # ------------------------------------------------------------------
    wm = LocalEnsembleWorldModel(
        model_kind="m1", model_path=world_model_path,
    )

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    all_results = {}  # method -> list of per-task dicts

    # ------------------------------------------------------------------
    # 1. RL search
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "#" * 60)
        print("# Phase 1: RL Search")
        print("#" * 60)

    rl_out_base = os.path.join(output_dir, "phase1_rl_design")
    rl_results = []

    for kind in task_kinds:
        model_dir = model_dirs.get(kind)
        if model_dir is None:
            if verbose:
                print(f"  SKIP {kind}: no model_dir provided")
            continue

        try:
            env, model, _tk, curve_keys = load_rl_model(model_dir, wm)
        except Exception as exc:
            if verbose:
                print(f"  SKIP {kind}: failed to load model -- {exc}")
            continue

        if verbose:
            print(f"\n  Task kind: {kind}  (curve_keys={len(curve_keys)})")

        for fc in fc_list:
            for bw in bw_list:
                tid = f"fc{fc:.0f}_bw{bw:.0f}"
                out_dir = os.path.join(rl_out_base, kind, tid)
                os.makedirs(out_dir, exist_ok=True)

                best_json = os.path.join(out_dir, "global_best.json")
                if os.path.exists(best_json):
                    with open(best_json) as fh:
                        rl_results.append(json.load(fh))
                    if verbose:
                        print(f"    {tid} SKIP (cached)")
                    continue

                if verbose:
                    print(f"    {tid} ...", end=" ", flush=True)
                t0 = time.perf_counter()
                res = rl_search(
                    env, model, fc, bw, curve_keys, n_episodes=n_evals,
                    verbose=False,
                )
                elapsed = time.perf_counter() - t0

                bp = res["pattern"]
                bp_items = res["pred_items"]
                br = res["best_reward"]
                n_metal = int(bp.sum())

                if verbose:
                    print(f"r={br:.4f} metal={n_metal}/100 ({elapsed:.1f}s)")

                # Save NPZ
                npz_path = _save_npz_results(
                    out_dir, bp, None, wm, pred_items=bp_items,
                )

                r = {
                    "method": "RL",
                    "task_kind": kind,
                    "task_id": tid,
                    "fc": float(fc),
                    "bw": float(bw),
                    "n_metal": n_metal,
                    "best_reward": float(br),
                    "n_episodes": int(n_evals),
                    "elapsed_sec": round(elapsed, 1),
                    "npz_path": npz_path,
                }
                json.dump(r, open(best_json, "w"), indent=2)
                rl_results.append(r)

    all_results["RL"] = rl_results
    if verbose:
        print(f"\n  RL done: {len(rl_results)} tasks")

    # ------------------------------------------------------------------
    # 2. GA + Random + DS search
    # ------------------------------------------------------------------
    for method in ["GA", "Random", "DS"]:
        if verbose:
            print(f"\n{'#' * 60}")
            print(f"# {method} Search")
            print(f"#{'#' * 60}")

        method_out = os.path.join(output_dir, f"phase2_{method.lower()}")
        method_results = []

        for kind in task_kinds:
            if verbose:
                print(f"\n  --- {kind} ---")

            for fc in fc_list:
                for bw in bw_list:
                    tid = f"fc{fc:.0f}_bw{bw:.0f}"
                    out_dir = os.path.join(method_out, kind, tid)
                    os.makedirs(out_dir, exist_ok=True)

                    best_json = os.path.join(out_dir, "best.json")
                    if os.path.exists(best_json):
                        with open(best_json) as fh:
                            method_results.append(json.load(fh))
                        if verbose:
                            print(f"    {tid} SKIP (cached)")
                        continue

                    # Build specs
                    specs, _ = build_active_design_task(
                        fc, bw, kind, "transmit", angle_pol=0,
                    )

                    t0 = time.perf_counter()

                    if method == "DS":
                        ds_res = search_dataset(fc, bw, kind)
                        best_pattern = ds_res["pattern"]
                        best_fit = ds_res["best_reward"]
                        n_queries_used = len(ds_res.get("all_rewards", []))
                    elif method == "GA":
                        best_pattern, best_fit, n_queries_used, _ = run_ga(
                            wm, specs, n_queries=n_evals,
                            pop_size=GA_POP,
                            tournament_k=GA_TOURNAMENT,
                            crossover_prob=GA_CROSSOVER,
                            mutation_prob=GA_MUTATION,
                            elitism=GA_ELITISM,
                            early_stop_stale=GA_EARLY_STOP,
                            verbose=False,
                        )
                    else:  # Random
                        best_pattern, best_fit, _ = run_random(
                            wm, specs, n_queries=n_evals, verbose=False,
                        )
                        n_queries_used = n_evals

                    elapsed = time.perf_counter() - t0
                    n_metal = int(best_pattern.sum())

                    if verbose:
                        print(
                            f"    {tid}: r={best_fit:.4f} "
                            f"metal={n_metal}/100 "
                            f"({n_queries_used}q, {elapsed:.1f}s)"
                        )

                    # Save NPZ
                    npz_path = _save_npz_results(
                        out_dir, best_pattern, specs, wm,
                    )

                    r = {
                        "method": method,
                        "task_kind": kind,
                        "task_id": tid,
                        "fc": float(fc),
                        "bw": float(bw),
                        "n_metal": n_metal,
                        "best_reward": float(best_fit),
                        "n_queries": n_queries_used,
                        "elapsed_sec": round(elapsed, 1),
                        "npz_path": npz_path,
                    }
                    json.dump(r, open(best_json, "w"), indent=2)
                    method_results.append(r)

        all_results[method] = method_results
        if verbose:
            print(f"\n  {method} done: {len(method_results)} tasks")

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 60}")
        print("Summary: Mean Reward by Method x Task Kind")
        print(f"{'=' * 60}")
        print(f"{'Method':<12} "
              f"{'Mode':>14} {'Angle':>14} {'Joint':>14} {'All':>14}")
        print("-" * 56)

    summary = {}
    for method in ["RL", "GA", "Random", "DS"]:
        method_data = all_results.get(method, [])
        parts = []
        all_vals = []

        for kind in task_kinds:
            # Gather rewards for this method+kind
            vals = [
                r["best_reward"]
                for r in method_data
                if r.get("task_kind") == kind
            ]
            if vals:
                parts.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}")
                all_vals.extend(vals)
            else:
                parts.append("—")

        if verbose and all_vals:
            print(
                f"{method:<12}  {'  '.join(parts)}  "
                f"{np.mean(all_vals):.4f}"
            )

        summary[method] = {
            "n_tasks": len(method_data),
            "mean_reward": float(np.mean(all_vals)) if all_vals else None,
            "std_reward": float(np.std(all_vals)) if all_vals else None,
            "by_kind": {},
        }
        for kind in task_kinds:
            vals = [
                r["best_reward"]
                for r in method_data
                if r.get("task_kind") == kind
            ]
            summary[method]["by_kind"][kind] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
            }

    # Save summary
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    json.dump(
        {"summary": summary, "config": {
            "fc_list": fc_list,
            "bw_list": bw_list,
            "task_kinds": task_kinds,
            "n_evals": n_evals,
        }},
        open(summary_path, "w"),
        indent=2,
        default=str,
    )

    if verbose:
        print(f"\nSummary saved: {summary_path}")
        print("Done.")

    return {
        "summary": summary,
        "results": all_results,
        "output_dir": output_dir,
    }
