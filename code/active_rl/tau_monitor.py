"""tau-monitoring wrapper around the active_rl training pipeline.

Behavior-identical to the plain CLI entry (active_rl.cli) — same env, same
reward, same gating logic, same PPO config. It ONLY adds instrumentation so a
tau sweep can be analyzed afterwards:

1.  gate_audit.jsonl   one record per uncertainty-gating evaluation:
                       U, surrogate reward, all 8 gate flags, decision, threshold.
                       Includes evaluations that were BLOCKED (interval /
                       duplicate pattern / below reward floor) so trigger-rate
                       vs tau can be reconstructed exactly.
2.  cst_calls.jsonl    one record per CST method invocation, including cache
                       hits and failures. actual_solves records the client
                       n_calls increment separately from design queries.
3.  monitor_meta.json  args dump, tau, seed, md5 of the loaded code files,
                       hostname, start/end wall time (even on crash).
4.  --seed             new: seeds random/numpy/torch and is injected into PPO
                       (SB3 BaseAlgorithm(seed=...) re-seeds all RNGs), so the
                       three tau arms share data/init/seed and only tau moves.
5.  candidates/        every 10k steps, save the best surrogate-scored design
                       in that window, plus up to 20 eligible untriggered
                       low-U controls. Each pair is JSON metadata and NPZ
                       pattern, pre-correction curves, and frequency grid.

Run it exactly like the plain trainer, e.g.:
    python active_rl/tau_monitor.py \
        --task-kind mode --design-mode transmit --angle-pol te \
        --world-model m1 --timesteps 100000 --cst-query-mode full \
        --selfplay-round-episodes 50 --cst-interval-episodes 10 \
        --min-surrogate-reward -1.5 \
        --uncertainty-threshold 0.05 --seed 42 \
        --world-model-path server/best_system_eq_robust_dropout_M1_c_theta0p0to60p0_order16.pth \
        --cst-host 127.0.0.1 --cst-port 65433 \
        --monitor-tag tau0p05_seed42

CST is ON by default (same as the trainer); pass --mock-cst for a no-hardware
smoke test or --no-cst to disable.

Monitor outputs go to results_monitor/<monitor-tag>/ (never into the
training run dir, so the training artifacts stay byte-compatible).
"""

import argparse
import hashlib
import json
import os
import random
import socket
import sys
import time
from datetime import datetime

import numpy as np
import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from active_rl import callbacks, cst_client, cst_gate, config, reward, trainer
from active_rl import cli
from active_rl.env import V11AngleDispersionEnv

MONITOR_ROOT = os.path.join(config.SCRIPT_DIR, "results_monitor")

MON = {
    "dir": None,
    "tag": "monitor",
    "t0": None,
    "seed": None,
    "gate_fh": None,
    "cst_fh": None,
    "cst_counter": 0,
    "cst_method_counter": 0,
    "gate_counter": 0,
    "latest_step": 0,
    "current_env": None,
    "window_best": None,
    "control_count": 0,
    "control_hashes": set(),
    "certified_best_r": float("-inf"),
    "certified_best": None,
    "certified_fh": None,
    "snapshot_interval": 10000,
    "control_limit": 20,
    "training_run_dir": None,
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _md5_of_file(path, limit_mb=64):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read(limit_mb * 1024 * 1024))
        return h.hexdigest()
    except Exception:
        return None


def _pattern_hash(arr):
    try:
        return hashlib.md5(np.asarray(arr, dtype=np.int8).tobytes()).hexdigest()[:12]
    except Exception:
        return None


def _now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def _append(fh_key, record):
    """Best-effort JSONL append; never raise into the training loop."""
    try:
        fh = MON[fh_key]
        if fh is None:
            names = {"gate_fh": "gate_audit.jsonl", "cst_fh": "cst_calls.jsonl",
                     "certified_fh": "certified_best.jsonl"}
            path = os.path.join(MON["dir"], names[fh_key])
            fh = open(path, "a", encoding="utf-8")
            MON[fh_key] = fh
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
    except Exception:
        pass


def _capture_candidate(env, pattern, specs, reward_value, uncertainty, episode_index,
                       triggered, kind):
    """Capture pre-CST, pre-finetune predictions for later independent validation."""
    geom = torch.from_numpy(np.asarray(pattern, dtype=np.float32)).to(config.DEVICE)
    geom = geom.unsqueeze(0).unsqueeze(0)
    curves = []
    with torch.no_grad():
        for spec in specs:
            pol = cst_gate._spec_pol_int(spec)
            s11, s21 = env.wm.predict(geom, theta=float(spec.get("theta", 0.0)), pol=pol)
            curves.append(np.stack((s11.detach().cpu().numpy().reshape(-1),
                                    s21.detach().cpu().numpy().reshape(-1))))
    return {
        "pattern": np.asarray(pattern, dtype=np.uint8).copy(),
        "predictions": np.asarray(curves, dtype=np.float32),
        "frequency_ghz": np.asarray(env.freqs_ghz, dtype=np.float32).copy(),
        "meta": {
            "kind": kind,
            "discovery_step": int(MON["latest_step"] + 1),
            "episode_index": int(episode_index),
            "specs": [dict(spec) for spec in specs],
            "predicted_reward": float(reward_value),
            "uncertainty": float(uncertainty),
            "triggered_online_cst": bool(triggered),
            "design_queries_before": int(env.query_state.n_design_queries),
            "pattern_hash": _pattern_hash(pattern),
            "prediction_stage": "before_online_cst_and_finetune",
        },
    }


def _save_candidate(candidate, stem, milestone_step=None):
    if candidate is None:
        return
    out_dir = os.path.join(MON["dir"], "candidates")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, stem + ".npz"),
                        pattern=candidate["pattern"],
                        predictions=candidate["predictions"],
                        frequency_ghz=candidate["frequency_ghz"])
    meta = dict(candidate["meta"])
    if milestone_step is not None:
        meta["milestone_step"] = int(milestone_step)
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _flush_window(step):
    _save_candidate(MON["window_best"], f"window_best_step{int(step):06d}", step)
    MON["window_best"] = None


# --------------------------------------------------------------------------
# patch 1: gating audit
# --------------------------------------------------------------------------
_orig_should_trigger = cst_gate.should_trigger_cst


def _should_trigger_audited(query_state, pattern_10x10, current_func_r, uncertainty,
                            episode_index, rng=None):
    trigger, reason = _orig_should_trigger(query_state, pattern_10x10, current_func_r,
                                           uncertainty, episode_index, rng=rng)
    MON["gate_counter"] += 1
    rec = {
        "t": _now_iso(),
        "eval_idx": MON["gate_counter"],
        "episode_index": int(episode_index),
        "uncertainty": float(uncertainty),
        "current_func_r": float(current_func_r),
        "best_surrogate_r": float(getattr(query_state, "best_surrogate_r", float("nan"))),
        "n_design_queries": int(getattr(query_state, "n_design_queries", -1)),
        "threshold": float(getattr(query_state, "uncertainty_threshold", float("nan"))),
        "pattern_hash": _pattern_hash(pattern_10x10),
        "trigger": bool(trigger),
        **{k: bool(v) for k, v in reason.items()},
    }
    _append("gate_fh", rec)
    env = MON["current_env"]
    if env is not None:
        best = MON["window_best"]
        if best is None or float(current_func_r) > best["meta"]["predicted_reward"]:
            MON["window_best"] = _capture_candidate(
                env, pattern_10x10, env.current_specs, current_func_r, uncertainty,
                episode_index, trigger, "window_best_predicted")
        # Deterministic, bounded sample of eligible low-U designs that the gate skipped.
        eligible_skip = (not trigger and not reason["interval_blocked"]
                         and not reason["already_seen"] and not reason["below_min_reward"]
                         and not reason["budget_exhausted"]
                         and not reason["predicted_new_best"]
                         and float(uncertainty) <= query_state.uncertainty_threshold)
        key = _pattern_hash(pattern_10x10)
        if (eligible_skip and key not in MON["control_hashes"]
                and MON["control_count"] < MON["control_limit"]
                and int(key, 16) % 10 == 0):
            candidate = _capture_candidate(
                env, pattern_10x10, env.current_specs, current_func_r, uncertainty,
                episode_index, False, "eligible_untriggered_control")
            MON["control_count"] += 1
            MON["control_hashes"].add(key)
            _save_candidate(candidate, f"control_{MON['control_count']:03d}")
    return trigger, reason


cst_gate.should_trigger_cst = _should_trigger_audited


# Keep the current environment available while the original step executes.
_orig_env_step = V11AngleDispersionEnv.step


def _step_with_context(self, action):
    MON["current_env"] = self
    try:
        return _orig_env_step(self, action)
    finally:
        MON["current_env"] = None


V11AngleDispersionEnv.step = _step_with_context


_orig_callback_step = callbacks.AngleVizCallback._on_step


def _on_step_with_snapshots(self):
    result = _orig_callback_step(self)
    MON["latest_step"] = int(self.num_timesteps)
    if self.num_timesteps % MON["snapshot_interval"] == 0:
        _flush_window(self.num_timesteps)
    return result


callbacks.AngleVizCallback._on_step = _on_step_with_snapshots


# --------------------------------------------------------------------------
# patch 2: CST call timing
# --------------------------------------------------------------------------
def _wrap_cst_class(cls):
    if cls is None or getattr(cls, "_mon_wrapped", False):
        return cls

    for meth in ("simulate", "simulate_both"):
        orig = getattr(cls, meth, None)
        if orig is None:
            continue

        def make_wrapped(f, meth_name):
            def wrapped(self, *args, **kwargs):
                t0 = time.perf_counter()
                before = int(getattr(self, "n_calls", 0))
                status = "ok"
                try:
                    return f(self, *args, **kwargs)
                except Exception:
                    status = "error"
                    raise
                finally:
                    solves = max(0, int(getattr(self, "n_calls", 0)) - before)
                    MON["cst_counter"] += solves
                    MON["cst_method_counter"] += 1
                    _append("cst_fh", {
                        "t": _now_iso(),
                        "method_idx": MON["cst_method_counter"],
                        "method": meth_name,
                        "status": status,
                        "actual_solves": solves,
                        "cumulative_actual_solves": MON["cst_counter"],
                        "wall_seconds": round(time.perf_counter() - t0, 3),
                        "pattern_hash": _pattern_hash(args[0]) if args else None,
                        "theta": args[1] if len(args) > 1 else kwargs.get("theta"),
                        "pol": args[2] if len(args) > 2 else kwargs.get("pol"),
                    })
            return wrapped

        setattr(cls, meth, make_wrapped(orig, meth))
    cls._mon_wrapped = True
    return cls


_wrap_cst_class(getattr(cst_client, "CSTClient", None))
_wrap_cst_class(getattr(cst_client, "MockCST", None))


# --------------------------------------------------------------------------
# patch 2b: certified best (CST-verified reward running max)
#
# The E8 audit showed best_surrogate_r freezes inflated early readings while
# CST-anchored honest readings stay far below. Record the honest counterpart
# at runtime: every CST query's aggregate TRUE reward (same aggregation the
# env itself uses), the running max, and per-query certified_best.jsonl rows.
# --------------------------------------------------------------------------
_orig_evaluate_cst_query_specs = cst_gate.evaluate_cst_query_specs


def _evaluate_cst_query_specs_certified(cst, wm, pattern_10x10, query_specs, freqs,
                                        add_to_buffer=True):
    rewards, specs, success = _orig_evaluate_cst_query_specs(
        cst, wm, pattern_10x10, query_specs, freqs, add_to_buffer=add_to_buffer)
    if rewards:
        try:
            env = MON["current_env"]
            weight = float(getattr(env, "stability_weight",
                                   config.DEFAULT_STABILITY_WEIGHT))
            agg = float(reward.aggregate_stability_reward(
                [float(r) for r in rewards], list(specs), weight))
            if agg > MON["certified_best_r"]:
                MON["certified_best_r"] = agg
                MON["certified_best"] = {
                    "agg_reward_cst": agg,
                    "discovery_step": int(MON["latest_step"]) + 1,
                    "pattern_hash": _pattern_hash(pattern_10x10),
                    "per_spec_rewards": [float(r) for r in rewards],
                    "specs": [dict(s) for s in specs],
                    "t": _now_iso(),
                }
            _append("certified_fh", {
                "t": _now_iso(),
                "agg_reward_cst": agg,
                "running_max": MON["certified_best_r"],
                "discovery_step": int(MON["latest_step"]) + 1,
                "pattern_hash": _pattern_hash(pattern_10x10),
                "per_spec_rewards": [float(r) for r in rewards],
                "n_specs": len(specs),
                "in_env": env is not None,
            })
        except Exception:
            pass  # instrumentation must never break training
    return rewards, specs, success


cst_gate.evaluate_cst_query_specs = _evaluate_cst_query_specs_certified


def _certified_best_meta():
    """certified-best block for monitor_meta.json (testable without training)."""
    return {
        "certified_best_r": (None if MON["certified_best_r"] == float("-inf")
                             else MON["certified_best_r"]),
        "certified_best": MON["certified_best"],
    }


# --------------------------------------------------------------------------
# patch 3: PPO seed injection
# --------------------------------------------------------------------------
_orig_PPO = trainer.PPO


def _PPO_seeded(*args, **kwargs):
    if MON["seed"] is not None:
        kwargs.setdefault("seed", int(MON["seed"]))
    return _orig_PPO(*args, **kwargs)


trainer.PPO = _PPO_seeded


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------
def _apply_target_sampling(rest, mode):
    """Map --target-sampling onto the target-noise mechanism.

    grid       -> unchanged rest (fixed 4x4 grid, the E8/September protocol).
    continuous -> force both noises to 0.1 by appending the flags LAST
    (argparse last-wins overrides any earlier occurrence, e.g. the sweep's
    explicit 0.0), reproducing the 183059 grid-centers +/- U(-0.1,+0.1) draw.
    """
    rest = list(rest)
    if mode == "continuous":
        rest += ["--target-fc-noise", "0.1", "--target-bw-noise", "0.1"]
    return rest


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--seed", type=int, default=None)
    pre.add_argument("--monitor-tag", default=None)
    pre.add_argument("--monitor-dir", default=None)
    pre.add_argument("--snapshot-interval", type=int, default=10000)
    pre.add_argument("--control-limit", type=int, default=20)
    pre.add_argument("--target-sampling", choices=["grid", "continuous"],
                     default="grid",
                     help="grid = fixed v11 4x4 (fc,bw) grid (default, E8 "
                          "protocol); continuous = grid centers +/- U(-0.1,+0.1) "
                          "GHz, the 183059 task distribution, mapped onto the "
                          "existing --target-fc-noise/--target-bw-noise flags")
    known, rest = pre.parse_known_args()
    rest = _apply_target_sampling(rest, known.target_sampling)

    # strip the monitor-only args from argv, then let the trainer CLI parse
    # the rest. IMPORTANT: cli.main/trainer.train install the v11 semantics
    # natively (fixed-target sampling, curve observations, terminal S-param
    # reward); there is no separate patch step to forget.
    sys.argv = [sys.argv[0]] + rest
    args = cli.parse_args()

    tag = known.monitor_tag or f"tau{args.uncertainty_threshold}_seed{known.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mon_dir = known.monitor_dir or os.path.join(MONITOR_ROOT, tag)
    os.makedirs(mon_dir, exist_ok=True)
    if known.snapshot_interval <= 0 or known.control_limit < 0:
        pre.error("snapshot-interval must be positive and control-limit nonnegative")
    if os.path.exists(mon_dir) and os.listdir(mon_dir):
        pre.error(f"monitor directory is not empty: {mon_dir}")
    MON.update(dir=mon_dir, tag=tag, seed=known.seed, t0=datetime.now(),
               snapshot_interval=known.snapshot_interval, control_limit=known.control_limit)

    # global seeding (covers env resets that happen before PPO(seed=...) re-seeds)
    if known.seed is not None:
        random.seed(known.seed)
        np.random.seed(known.seed)
        torch.manual_seed(known.seed)

    print(f"[tau-monitor] tag={tag} tau={args.uncertainty_threshold} seed={known.seed} "
          f"sampling={known.target_sampling}")
    print(f"[tau-monitor] logs -> {mon_dir}")

    t0 = time.perf_counter()
    status = "ok"
    try:
        MON["training_run_dir"] = trainer.train(args)
        if MON["window_best"] is not None:
            _flush_window(args.timesteps)
    except Exception as exc:  # still write meta on crash
        status = f"crash: {exc!r}"
        raise
    finally:
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        code_files = [
            __file__,
            os.path.join(_pkg_dir, "config.py"),
            os.path.join(_pkg_dir, "networks.py"),
            os.path.join(_pkg_dir, "cst_client.py"),
            os.path.join(_pkg_dir, "reward.py"),
            os.path.join(_pkg_dir, "tasks.py"),
            os.path.join(_pkg_dir, "world_model.py"),
            os.path.join(_pkg_dir, "cst_gate.py"),
            os.path.join(_pkg_dir, "env.py"),
            os.path.join(_pkg_dir, "features.py"),
            os.path.join(_pkg_dir, "callbacks.py"),
            os.path.join(_pkg_dir, "eval_sweep.py"),
            os.path.join(_pkg_dir, "trainer.py"),
            os.path.join(_pkg_dir, "cli.py"),
        ]
        # Reconcile CST counts with the training run's own bookkeeping:
        #   cst_calls        = client n_calls delta (actual server requests)
        #   cst_method_calls = wrapped simulate/simulate_both invocations,
        #                      including cache hits
        #   run_meta value   = what the training run itself recorded
        run_meta = None
        if MON.get("training_run_dir"):
            try:
                with open(os.path.join(MON["training_run_dir"], "run_meta.json"),
                          "r", encoding="utf-8") as f:
                    run_meta = json.load(f)
            except Exception:
                run_meta = None
        meta = {
            "monitor_tag": tag,
            "status": status,
            "tau": float(args.uncertainty_threshold),
            "target_sampling": known.target_sampling,
            "seed": known.seed,
            "t_start": MON["t0"].isoformat(),
            "t_end": datetime.now().isoformat(),
            "wall_hours": round((time.perf_counter() - t0) / 3600.0, 3),
            "gate_evaluations": MON["gate_counter"],
            "cst_calls": MON["cst_counter"],
            "cst_method_calls": MON["cst_method_counter"],
            "cst_calls_reconcile": {
                "client_n_calls": MON["cst_counter"],
                "method_calls_incl_cache_hits": MON["cst_method_counter"],
                "run_meta_cst_calls": (run_meta or {}).get(
                    "actual_cst_calls", (run_meta or {}).get("cst_calls")),
                "match": ((run_meta or {}).get(
                    "actual_cst_calls", (run_meta or {}).get("cst_calls"))
                    == MON["cst_counter"]),
            },
            "saved_controls": MON["control_count"],
            **_certified_best_meta(),
            "snapshot_interval": MON["snapshot_interval"],
            "training_run_dir": MON["training_run_dir"],
            "training_run_meta": run_meta,
            "hostname": socket.gethostname(),
            "v9_output_dir": config.RESULTS_DIR,
            "code_md5": {os.path.basename(p): _md5_of_file(p) for p in code_files},
            "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                     for k, v in vars(args).items()},
        }
        with open(os.path.join(mon_dir, "monitor_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        for key in ("gate_fh", "cst_fh", "certified_fh"):
            try:
                if MON[key] is not None:
                    MON[key].close()
                    MON[key] = None
            except Exception:
                pass
        print(f"[tau-monitor] done status={status} "
              f"gate_evals={MON['gate_counter']} cst_calls={MON['cst_counter']} -> {mon_dir}")


if __name__ == "__main__":
    main()
