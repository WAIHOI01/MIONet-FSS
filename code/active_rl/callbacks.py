"""Training callback: TensorBoard logging, cycle metrics, design snapshots.

matplotlib backend and rcParams replicate the original v8 base imports so the
snapshot/sweep PNGs keep their exact appearance.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "grid.linestyle": ":"})
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback

from active_rl.config import DEVICE, DEFAULT_STABILITY_WEIGHT, STANDARD_FREQS
from active_rl.cst_gate import (
    select_representative_specs,
    get_cst_call_count,
    cst_budget_label,
)


def pattern_for_cst_display(pattern):
    """Return display image with CST x/y axes and dark metal on light background."""
    return 1.0 - np.asarray(pattern, dtype=np.float32).T
def save_window_best_artifacts(out_png, step, fc_0, bw_0, reward, quarter,
                                pattern_10x10, specs, pred_items, te_type=None,
                                stability_weight=DEFAULT_STABILITY_WEIGHT):
    out_png = Path(out_png)
    quarter = np.asarray(quarter, dtype=np.float32)
    pattern_10x10 = np.asarray(pattern_10x10, dtype=np.float32)

    predictions_meta = []
    npz_payload = {
        "quarter_5x5": quarter,
        "pattern_10x10": pattern_10x10,
        "freqs_ghz": STANDARD_FREQS.astype(np.float32),
    }
    for item in pred_items:
        pol = int(item["pol"])
        theta = float(item.get("theta", 0.0))
        theta_key = int(round(theta))
        s11 = np.asarray(item["s11"], dtype=np.float32)
        s21 = np.asarray(item["s21"], dtype=np.float32)
        predictions_meta.append({
            "label": item["label"], "type": item["type"],
            "pol": pol, "theta": theta,
        })
        npz_payload[f"pred_s11_pol{pol}_theta{theta_key}"] = s11
        npz_payload[f"pred_s21_pol{pol}_theta{theta_key}"] = s21

    meta = {
        "step": int(step), "fc_0": float(fc_0), "bw_0": float(bw_0),
        "te_type": te_type,
        "stability_weight": float(stability_weight),
        "reward": float(reward),
        "specs": [dict(spec) for spec in specs],
        "predictions": predictions_meta,
    }
    out_png.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(out_png.with_suffix(".npz"), **npz_payload)


def is_terminal_design_info(info):
    return bool(info.get("design_complete", False)) and "episode_reward" in info


def extract_cst_tensorboard_metrics(info):
    return {
        "trigger": 1.0 if bool(info.get("trigger_cst", False)) else 0.0,
        "sim_calls": float(info.get("cst_calls", 0.0) or 0.0),
        "design_queries": float(info.get("cst_design_queries", 0.0) or 0.0),
        "success": float(info.get("cst_success", 0.0) or 0.0),
    }


def summarize_cycle_rewards(rewards):
    values = np.asarray([float(r) for r in rewards], dtype=np.float32)
    if values.size == 0:
        return {
            "sample_count": 0,
            "mean_reward": float("nan"),
            "std_reward": float("nan"),
            "min_reward": float("nan"),
            "max_reward": float("nan"),
        }
    return {
        "sample_count": int(values.size),
        "mean_reward": float(values.mean()),
        "std_reward": float(values.std()),
        "min_reward": float(values.min()),
        "max_reward": float(values.max()),
    }


def _json_finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def format_cycle_progress(round_index, episode_count, timestep, round_best_reward,
                          global_best_reward, cst_client=None, query_state=None,
                          cycle_sample_count=None, cycle_mean_reward=None):
    design_queries = int(getattr(query_state, "n_design_queries", 0) if query_state is not None else 0)
    round_best = float(round_best_reward) if np.isfinite(round_best_reward) else float("nan")
    global_best = float(global_best_reward) if np.isfinite(global_best_reward) else float("nan")
    sample_text = ""
    if cycle_sample_count is not None:
        sample_text += f" samples={int(cycle_sample_count)}"
    if cycle_mean_reward is not None:
        mean_reward = float(cycle_mean_reward) if np.isfinite(cycle_mean_reward) else float("nan")
        sample_text += f" mean_reward={mean_reward:.4f}"
    return (
        f"[cycle {int(round_index)}] episodes={int(episode_count)} steps={int(timestep)}"
        f"{sample_text} "
        f"round_best={round_best:.4f} global_best={global_best:.4f} "
        f"cst_calls={get_cst_call_count(cst_client)} queries={design_queries}/{cst_budget_label(query_state)}"
    )


class AngleVizCallback(BaseCallback):
    def __init__(self, run_dir, round_episodes=50, verbose=0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.round_episodes = max(1, int(round_episodes))
        self.tb_dir = os.path.join(run_dir, "tb")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        self.episode_count = 0
        self.round_index = 1
        self.round_best_reward = -float("inf")
        self.round_best_M = None
        self.round_best_quarter = None
        self.round_best_specs = None
        self.round_best_fc_0 = None
        self.round_best_bw_0 = None
        self.round_best_te_type = None
        self.round_rewards = []
        self.best_mean_reward = -float("inf")
        self.best_mean_reward_cycle = None
        self.best_mean_reward_stats = None
        self.cycle_history = []
        self.last_saved_global_best_reward = -float("inf")

    def _reset_round(self):
        self.round_index += 1
        self.round_best_reward = -float("inf")
        self.round_best_M = None
        self.round_best_quarter = None
        self.round_best_specs = None
        self.round_best_fc_0 = None
        self.round_best_bw_0 = None
        self.round_best_te_type = None
        self.round_rewards = []

    def _capture_terminal_info(self, info):
        M = torch.tensor(info["terminal_pattern_10x10"], dtype=torch.float32, device=DEVICE)
        quarter = torch.tensor(info["terminal_quarter_5x5"], dtype=torch.float32, device=DEVICE)
        specs = [dict(spec) for spec in info.get("terminal_specs", [])]
        return {
            "M": M,
            "quarter": quarter,
            "specs": specs,
            "fc_0": float(info.get("terminal_fc_0", info.get("fc_0", 0.0))),
            "bw_0": float(info.get("terminal_bw_0", info.get("bw_0", 0.0))),
            "te_type": str(info.get("terminal_te_type", info.get("te_type", "unknown"))),
            "reward": float(info["episode_reward"]),
        }

    def _on_step(self):
        env0 = self.training_env.envs[0]
        self.writer.add_scalar("train/global_best_reward", float(env0.global_best_reward), self.num_timesteps)
        self.writer.add_scalar("train/episode_best_reward", float(env0.best_reward), self.num_timesteps)
        self.writer.add_scalar("train/uncertainty", float(getattr(env0, "last_uncertainty", 0.0)), self.num_timesteps)
        round_scalar = self.round_best_reward if np.isfinite(self.round_best_reward) else float(getattr(env0, "last_func_r", -float("inf")))
        self.writer.add_scalar("train/round_best_reward", float(round_scalar), self.num_timesteps)
        if env0.current_fc_0 is not None:
            self.writer.add_scalar("target/fc_0", float(env0.current_fc_0), self.num_timesteps)
        if env0.current_bw_0 is not None:
            self.writer.add_scalar("target/bw_0", float(env0.current_bw_0), self.num_timesteps)
        cst_client = getattr(env0, "cst", None)
        query_state = getattr(env0, "query_state", None)
        self.writer.add_scalar("cst/sim_calls", float(get_cst_call_count(cst_client)), self.num_timesteps)
        self.writer.add_scalar(
            "cst/design_queries",
            float(getattr(query_state, "n_design_queries", 0) if query_state is not None else 0),
            self.num_timesteps,
        )
        self.writer.add_scalar(
            "cst/budget",
            float(getattr(query_state, "cst_budget", 0) if query_state is not None else 0),
            self.num_timesteps,
        )

        infos = self.locals.get("infos", []) if hasattr(self, "locals") else []
        for info in infos:
            if not is_terminal_design_info(info):
                continue
            self.episode_count += 1
            payload = self._capture_terminal_info(info)
            ep_reward = payload["reward"]
            self.round_rewards.append(float(ep_reward))
            self.writer.add_scalar("episode/reward", ep_reward, self.episode_count)
            self.writer.add_scalar("episode/round_index", self.round_index, self.episode_count)
            cst_metrics = extract_cst_tensorboard_metrics(info)
            self.writer.add_scalar("episode/cst_trigger", cst_metrics["trigger"], self.episode_count)
            self.writer.add_scalar("episode/cst_sim_calls", cst_metrics["sim_calls"], self.episode_count)
            self.writer.add_scalar("episode/cst_design_queries", cst_metrics["design_queries"], self.episode_count)
            self.writer.add_scalar("episode/cst_success", cst_metrics["success"], self.episode_count)
            self.writer.add_scalar("cst/trigger_event", cst_metrics["trigger"], self.num_timesteps)
            if cst_metrics["sim_calls"] > 0:
                self.writer.add_scalar("cst/terminal_sim_calls", cst_metrics["sim_calls"], self.num_timesteps)
            if cst_metrics["design_queries"] > 0:
                self.writer.add_scalar("cst/terminal_design_queries", cst_metrics["design_queries"], self.num_timesteps)
            if cst_metrics["success"] > 0:
                self.writer.add_scalar("cst/success_per_query", cst_metrics["success"], self.num_timesteps)

            if ep_reward > self.round_best_reward:
                self.round_best_reward = ep_reward
                self.round_best_M = payload["M"]
                self.round_best_quarter = payload["quarter"]
                self.round_best_specs = payload["specs"]
                self.round_best_fc_0 = payload["fc_0"]
                self.round_best_bw_0 = payload["bw_0"]
                self.round_best_te_type = payload["te_type"]

            if ep_reward > self.last_saved_global_best_reward:
                self._save_named_snapshot(
                    "global_best", payload["M"], payload["quarter"], payload["specs"],
                    payload["fc_0"], payload["bw_0"], ep_reward,
                    te_type=payload["te_type"])
                self.last_saved_global_best_reward = ep_reward

            if self.episode_count % self.round_episodes == 0:
                self._save_round_snapshot()
                self._reset_round()
        return True

    def _snapshot_path(self, prefix):
        if prefix == "global_best":
            return os.path.join(self.run_dir, "global_best.png")
        if prefix.startswith("round_"):
            return os.path.join(self.run_dir, f"{prefix}.png")
        return os.path.join(self.run_dir, f"{prefix}_{self.n_calls:06d}.png")

    def _save_named_snapshot(self, prefix, M, quarter, specs, fc_0, bw_0, snap_reward, te_type=None):
        env0 = self.training_env.envs[0]
        wm = env0.wm
        if M is None or quarter is None or not specs:
            return
        if not torch.is_tensor(M):
            M = torch.tensor(M, dtype=torch.float32, device=DEVICE)
        if not torch.is_tensor(quarter):
            quarter = torch.tensor(quarter, dtype=torch.float32, device=DEVICE)

        key_specs = select_representative_specs(specs)
        n_spec = min(4, len(key_specs))
        fig, axes = plt.subplots(1, n_spec + 2, figsize=(4.0 * (n_spec + 2), 4))
        if n_spec + 2 == 1:
            axes = [axes]
        Mt = M.unsqueeze(0).unsqueeze(0)
        pred_items = []
        for idx, spec in enumerate(key_specs[:n_spec]):
            p11, p21 = wm.predict(Mt, spec.get("theta", 0.0), spec.get("pol", 0))
            p11 = p11.detach().cpu().numpy().flatten()
            p21 = p21.detach().cpu().numpy().flatten()
            pred_items.append({
                "label": spec["label"], "type": spec["type"],
                "pol": int(spec.get("pol", 0)),
                "theta": float(spec.get("theta", 0.0)),
                "s11": p11, "s21": p21,
            })
            eps = 1e-8
            axes[idx].plot(env0.freqs_ghz, 20 * np.log10(np.abs(p11) + eps), "#1f77b4", lw=2.0, label="S11")
            axes[idx].plot(env0.freqs_ghz, 20 * np.log10(np.abs(p21) + eps), "#d62728", lw=2.0, label="S21")
            axes[idx].axvspan(spec["fc"] - spec["bw"] / 2, spec["fc"] + spec["bw"] / 2,
                              alpha=0.12, color="green")
            axes[idx].set_title(spec["label"], fontsize=9)
            axes[idx].set_ylim(-40, 5)
            axes[idx].set_xlim(float(env0.freqs_ghz[0]), float(env0.freqs_ghz[-1]))
            axes[idx].grid(True, alpha=0.25)
            axes[idx].legend(fontsize=7)

        quarter_ax = axes[n_spec]
        full_ax = axes[n_spec + 1]
        quarter_ax.imshow(pattern_for_cst_display(quarter.detach().cpu().numpy()), cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        quarter_active = int(torch.count_nonzero(quarter > 0.5).item())
        full_active = int(torch.count_nonzero(M > 0.5).item())
        quarter_ax.set_title(f"quarter 5x5\\nactive={quarter_active}/25", fontsize=9)
        quarter_ax.set_xticks([])
        quarter_ax.set_yticks([])

        full_ax.imshow(pattern_for_cst_display(M.detach().cpu().numpy()), cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        full_ax.set_title(
            f"{prefix}\nep {self.episode_count} step {self.num_timesteps}\n"
            f"fc0={fc_0:.2f} bw0={bw_0:.2f}\nr={snap_reward:.3f}",
            fontsize=9)
        full_ax.set_xticks([])
        full_ax.set_yticks([])
        fig.tight_layout()
        out_path = self._snapshot_path(prefix)
        fig.savefig(out_path, dpi=140)
        save_window_best_artifacts(
            out_png=out_path, step=self.num_timesteps, fc_0=fc_0, bw_0=bw_0,
            reward=snap_reward,
            quarter=quarter.detach().cpu().numpy(),
            pattern_10x10=M.detach().cpu().numpy(),
            specs=specs, pred_items=pred_items,
            te_type=te_type,
            stability_weight=getattr(env0, "stability_weight", DEFAULT_STABILITY_WEIGHT))
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        self.writer.add_image(f"design/{prefix}", img.transpose(2, 0, 1), self.num_timesteps)
        self.writer.add_text(
            f"design/{prefix}_meta",
            f"episode={self.episode_count}, step={self.num_timesteps}, fc0={fc_0:.4f}, "
            f"bw0={bw_0:.4f}, reward={snap_reward:.6f}",
            self.num_timesteps)
        self.writer.flush()
        plt.close(fig)

    def _cycle_stats_record(self, stats):
        return {
            "cycle": int(self.round_index),
            "episode_count": int(self.episode_count),
            "timesteps": int(self.num_timesteps),
            "sample_count": int(stats["sample_count"]),
            "mean_reward": _json_finite_or_none(stats["mean_reward"]),
            "std_reward": _json_finite_or_none(stats["std_reward"]),
            "min_reward": _json_finite_or_none(stats["min_reward"]),
            "max_reward": _json_finite_or_none(stats["max_reward"]),
            "round_best_reward": _json_finite_or_none(self.round_best_reward),
            "best_mean_reward": _json_finite_or_none(self.best_mean_reward),
        }

    def _append_cycle_metrics(self, stats):
        record = self._cycle_stats_record(stats)
        self.cycle_history.append(record)
        with open(os.path.join(self.run_dir, "cycle_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def _maybe_save_best_mean_reward_model(self, stats):
        sample_count = int(stats["sample_count"])
        mean_reward = float(stats["mean_reward"])
        if sample_count <= 0 or not np.isfinite(mean_reward):
            return False
        if mean_reward <= self.best_mean_reward:
            return False
        self.best_mean_reward = mean_reward
        self.best_mean_reward_cycle = int(self.round_index)
        self.best_mean_reward_stats = dict(stats)
        model_path = os.path.join(self.run_dir, "ppo_best_mean_reward")
        self.model.save(model_path)
        meta = {
            "cycle": int(self.round_index),
            "episode_count": int(self.episode_count),
            "timesteps": int(self.num_timesteps),
            "sample_count": sample_count,
            "mean_reward": float(mean_reward),
            "std_reward": _json_finite_or_none(stats["std_reward"]),
            "min_reward": _json_finite_or_none(stats["min_reward"]),
            "max_reward": _json_finite_or_none(stats["max_reward"]),
            "model_path": model_path + ".zip",
        }
        with open(os.path.join(self.run_dir, "best_mean_reward.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(
            f"[cycle] saved ppo_best_mean_reward: cycle={self.round_index} "
            f"samples={sample_count} mean_reward={mean_reward:.4f}"
        )
        return True

    def _save_round_snapshot(self):
        env0 = self.training_env.envs[0]
        cycle_stats = summarize_cycle_rewards(self.round_rewards)
        progress = format_cycle_progress(
            round_index=self.round_index,
            episode_count=self.episode_count,
            timestep=self.num_timesteps,
            round_best_reward=self.round_best_reward,
            global_best_reward=getattr(env0, "global_best_reward", -float("inf")),
            cst_client=getattr(env0, "cst", None),
            query_state=getattr(env0, "query_state", None),
            cycle_sample_count=cycle_stats["sample_count"],
            cycle_mean_reward=cycle_stats["mean_reward"],
        )
        print(progress)
        self.writer.add_text("cycle/progress", progress, self.num_timesteps)
        self._maybe_save_best_mean_reward_model(cycle_stats)
        self._append_cycle_metrics(cycle_stats)
        cycle_round_best = self.round_best_reward if np.isfinite(self.round_best_reward) else float(getattr(env0, "last_func_r", 0.0))
        cycle_global_best = float(getattr(env0, "global_best_reward", -float("inf")))
        if not np.isfinite(cycle_global_best):
            cycle_global_best = cycle_round_best
        self.writer.add_scalar("cycle/sample_count", float(cycle_stats["sample_count"]), self.round_index)
        if np.isfinite(cycle_stats["mean_reward"]):
            self.writer.add_scalar("cycle/mean_reward", float(cycle_stats["mean_reward"]), self.round_index)
            self.writer.add_scalar("cycle/std_reward", float(cycle_stats["std_reward"]), self.round_index)
            self.writer.add_scalar("cycle/min_reward", float(cycle_stats["min_reward"]), self.round_index)
            self.writer.add_scalar("cycle/max_reward", float(cycle_stats["max_reward"]), self.round_index)
            self.writer.add_scalar("cycle/best_mean_reward", float(self.best_mean_reward), self.round_index)
        self.writer.add_scalar("cycle/round_best_reward", float(cycle_round_best), self.round_index)
        self.writer.add_scalar("cycle/global_best_reward", float(cycle_global_best), self.round_index)
        self.writer.add_scalar("cycle/cst_sim_calls", float(get_cst_call_count(getattr(env0, "cst", None))), self.round_index)
        query_state = getattr(env0, "query_state", None)
        self.writer.add_scalar(
            "cycle/cst_design_queries",
            float(getattr(query_state, "n_design_queries", 0) if query_state is not None else 0),
            self.round_index,
        )
        if self.round_best_M is None or self.round_best_quarter is None or not self.round_best_specs:
            self.writer.flush()
            return
        prefix = f"round_{self.round_index:04d}_best"
        self._save_named_snapshot(
            prefix, self.round_best_M, self.round_best_quarter, self.round_best_specs,
            self.round_best_fc_0, self.round_best_bw_0, self.round_best_reward,
            te_type=self.round_best_te_type)
        print(f"[round] saved {prefix}: episodes={self.episode_count}, reward={self.round_best_reward:.4f}")

    def _on_training_end(self):
        if self.round_rewards or self.round_best_M is not None:
            self._save_round_snapshot()
        env0 = self.training_env.envs[0]
        if env0.global_best_M is not None and env0.global_best_quarter is not None and env0.global_best_specs:
            self._save_named_snapshot(
                "global_best", env0.global_best_M, env0.global_best_quarter,
                env0.global_best_specs, env0.global_best_fc_0, env0.global_best_bw_0,
                env0.global_best_reward, te_type=env0.global_best_te_type)
        self.writer.flush()
        self.writer.close()
