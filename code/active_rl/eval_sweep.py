"""Post-training policy rollout and (fc, bw) parameter sweep.

evaluate_cst_query_specs is reached via the cst_gate module so the tau
monitor's certified-best wrapper keeps working.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "grid.linestyle": ":"})
import numpy as np

from active_rl import cst_gate
from active_rl.config import (
    STANDARD_FREQS,
    SWEEP_FC_LIST,
    SWEEP_BW_LIST,
    SWEEP_DESIGN_TYPES,
    DEFAULT_STABILITY_WEIGHT,
)
from active_rl.reward import aggregate_stability_reward, task_reward_v11
from active_rl.env import summarize_uncertainty
from active_rl.tasks import build_active_design_task


def run_policy_episode(model, env, deterministic=True):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    M = env.best_M if env.best_M is not None else env.current_M
    pattern_10x10 = M.detach().cpu().numpy().astype(np.float32)
    pred_items = []
    rewards = []
    for spec in env.current_specs:
        s11, s21 = env.wm.predict(M.unsqueeze(0).unsqueeze(0), spec["theta"], spec["pol"])
        s11 = s11.detach().cpu().numpy().flatten()
        s21 = s21.detach().cpu().numpy().flatten()
        spec_reward, _, _ = task_reward_v11(s11, s21, env.freqs_ghz, spec)
        rewards.append(spec_reward)
        pred_items.append({
            "label": spec["label"], "type": spec["type"], "pol": int(spec["pol"]),
            "theta": float(spec["theta"]), "fc": spec["fc"], "bw": spec["bw"],
            "s11": s11, "s21": s21,
        })
    score = aggregate_stability_reward(
        rewards, env.current_specs, getattr(env, "stability_weight", DEFAULT_STABILITY_WEIGHT))
    return pattern_10x10, pred_items, float(score)

def plot_sweep_summary(results, sweep_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    xs = [r["fc_0"] for r in results]
    ys = [r["bw_0"] for r in results]
    cs = [r["surrogate_reward"] for r in results]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=140, edgecolors="black")
    ax.set_xlabel("fc_0 (GHz)"); ax.set_ylabel("bw_0 (GHz)")
    ax.set_title("Sweep: active_design")
    fig.colorbar(sc, ax=ax, label="Avg surrogate reward")
    fig.tight_layout()
    fig.savefig(os.path.join(sweep_dir, "sweep_summary.png"), dpi=140)
    plt.close(fig)


def sweep_te_types(design_mode):
    if design_mode == "both":
        return list(SWEEP_DESIGN_TYPES)
    if design_mode in SWEEP_DESIGN_TYPES:
        return [design_mode]
    raise ValueError(f"Unknown design_mode: {design_mode}")

def run_parameter_sweep(model, env, cst, sweep_dir, verify_top_n_cst=4):
    os.makedirs(sweep_dir, exist_ok=True)
    results = []
    for fc_0 in SWEEP_FC_LIST:
        for bw_0 in SWEEP_BW_LIST:
            for te_type in sweep_te_types(env.design_mode):
                env.set_target(fc_0, bw_0, te_type)
                pattern_10x10, pred_items, surrogate_reward = run_policy_episode(
                    model, env, deterministic=True)
                uncertainty = summarize_uncertainty(env.wm, pattern_10x10, env.current_specs)
                results.append({
                    "fc_0": fc_0, "bw_0": bw_0, "te_type": te_type,
                    "surrogate_reward": surrogate_reward,
                    "uncertainty": float(uncertainty),
                    "pattern_10x10": pattern_10x10.tolist(),
                    "pred_items": pred_items,
                })
                print(f"  sweep fc0={fc_0:>4.1f} bw0={bw_0:>3.1f} te={te_type:<8s} "
                      f"reward={surrogate_reward:>8.4f} unc={uncertainty:>6.3f}")
    env.clear_target_override()

    sorted_results = sorted(results, key=lambda x: x["surrogate_reward"], reverse=True)
    for entry in sorted_results[:verify_top_n_cst]:
        pattern = np.array(entry["pattern_10x10"], dtype=np.float32)
        specs, _ = build_active_design_task(
            entry["fc_0"], entry["bw_0"], env.task_kind, entry["te_type"], angle_pol=env.angle_pol)
        try:
            cst_rewards, cst_specs, _ = cst_gate.evaluate_cst_query_specs(
                cst, None, pattern, specs, STANDARD_FREQS, add_to_buffer=False)
            entry["cst_reward"] = float(aggregate_stability_reward(
                cst_rewards, cst_specs, getattr(env, "stability_weight", DEFAULT_STABILITY_WEIGHT)))
            entry["cst_per_spec"] = {
                spec["label"]: float(r) for spec, r in zip(cst_specs, cst_rewards)}
        except Exception as exc:
            entry["cst_error"] = str(exc)

    with open(os.path.join(sweep_dir, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2)
    plot_sweep_summary(sorted_results, sweep_dir)
    return sorted_results
