"""PPO training entry: rollout geometry, env/PPO construction, run artifacts.

`PPO` is imported at module level on purpose — the tau monitor wraps
trainer.PPO to inject the seed. The v11 semantics (terminal S-parameter
reward, fixed-target grid sampling, curve observations) are native here;
trainer.train() ends by augmenting run_meta.json with the v11 keys.
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from active_rl import tasks
from active_rl.callbacks import AngleVizCallback, _json_finite_or_none
from active_rl.config import DEVICE, RESULTS_DIR
from active_rl.cst_client import CSTClient, MockCST
from active_rl.cst_gate import ActiveQueryState, get_cst_call_count
from active_rl.env import V11AngleDispersionEnv
from active_rl.eval_sweep import run_parameter_sweep
from active_rl.features import V11AngleDispersionFeatures
from active_rl.world_model import LocalEnsembleWorldModel


def compute_selfplay_rollout_steps(episode_len, round_episodes):
    episode_len = max(1, int(episode_len))
    round_episodes = max(1, int(round_episodes))
    return episode_len * round_episodes


def augment_v11_run_meta(run_dir, args):
    """Add v11-specific fixed-target metadata after run_meta.json is written."""
    meta_path = os.path.join(str(run_dir), "run_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.update({
        "rl_version": "v11",
        "reward_version": "terminal_sparam_inband_minus_outband",
        "observation_target_masks": list(tasks.V11_TARGET_KEYS),
        "observation_sparam_angles": list(tasks.V11_CURVE_ANGLES),
        "observation_sparams": list(tasks.V11_CURVE_KEYS),
        "target_fc_list": list(tasks.V11_FC_LIST),
        "target_bw_list": list(tasks.V11_BW_LIST),
        "target_fc_noise": float(getattr(args, "target_fc_noise", tasks.TARGET_FC_NOISE)),
        "target_bw_noise": float(getattr(args, "target_bw_noise", tasks.TARGET_BW_NOISE)),
        "target_sampling": str(getattr(args, "target_sampling", tasks.TARGET_SAMPLING)),
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def train(args):
    # v11 fixed-target sampling globals must be set before the env is built
    # (the env reads them at construction time, same as the original v11).
    tasks.TARGET_FC_NOISE = float(getattr(args, "target_fc_noise", 0.0))
    tasks.TARGET_BW_NOISE = float(getattr(args, "target_bw_noise", 0.0))
    tasks.TARGET_SAMPLING = str(getattr(args, "target_sampling", "grid"))

    label = f"{args.task_kind}_{args.design_mode}"
    if getattr(args, "tag", None):
        label += f"_{args.tag}"
    if getattr(args, "seed", None) is not None:
        label += f"_s{args.seed}"
        # Seed before env creation so env-side sampling is reproducible too.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    label += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(run_dir, exist_ok=True)

    wm = LocalEnsembleWorldModel(model_kind=args.world_model, model_path=args.world_model_path)
    cst = None if args.no_cst else (MockCST(wm) if args.mock_cst else CSTClient(args.cst_host, args.cst_port))
    query_state = ActiveQueryState(
        cst_budget=args.cst_budget,
        cst_interval_episodes=args.cst_interval_episodes,
        min_surrogate_reward=args.min_surrogate_reward,
        uncertainty_threshold=args.uncertainty_threshold,
    )
    env_factory = lambda: V11AngleDispersionEnv(
        wm=wm, cst_client=cst, episode_len=args.episode_len, design_mode=args.design_mode,
        stability_weight=args.stability_weight, active_query_state=query_state,
        cst_query_mode=args.cst_query_mode, task_kind=args.task_kind,
        angle_pol=args.angle_pol,
        min_active_quarter_pixels=args.min_active_quarter_pixels,
        invalid_pattern_reward=args.invalid_pattern_reward)
    vec_env = DummyVecEnv([env_factory])

    policy_kwargs = dict(
        features_extractor_class=V11AngleDispersionFeatures,
        features_extractor_kwargs=dict(dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    model = PPO(
        "MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate, n_steps=compute_selfplay_rollout_steps(args.episode_len, args.selfplay_round_episodes), batch_size=64, n_epochs=10,
        gamma=1.0, gae_lambda=1.0, clip_range=args.clip_range, ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1, device=DEVICE,
    )

    callback = AngleVizCallback(run_dir=run_dir, round_episodes=args.selfplay_round_episodes)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(os.path.join(run_dir, "ppo_final"))
    if hasattr(wm, "save_checkpoint"):
        wm.save_checkpoint(os.path.join(run_dir, "world_model_final.pth"))
    if args.sweep:
        sweep_dir = os.path.join(run_dir, "sweep")
        sweep_env = V11AngleDispersionEnv(
            wm=wm, cst_client=cst, episode_len=args.episode_len,
            design_mode=args.design_mode,
            stability_weight=args.stability_weight, active_query_state=query_state,
            cst_query_mode=args.cst_query_mode, task_kind=args.task_kind,
            angle_pol=args.angle_pol,
            min_active_quarter_pixels=args.min_active_quarter_pixels,
            invalid_pattern_reward=args.invalid_pattern_reward)
        run_parameter_sweep(model, sweep_env, cst, sweep_dir=sweep_dir,
                            verify_top_n_cst=(0 if cst is None else args.sweep_top_n))

    run_meta = {
        "task_kind": args.task_kind,
        "design_mode": args.design_mode,
        "angle_pol": args.angle_pol,
        "stability_weight": args.stability_weight,
        "world_model": args.world_model,
        "world_model_path": getattr(wm, "model_path", args.world_model_path),
        "cst_budget": args.cst_budget,
        "cst_interval_episodes": args.cst_interval_episodes,
        "cst_query_mode": args.cst_query_mode,
        "min_active_quarter_pixels": args.min_active_quarter_pixels,
        "invalid_pattern_reward": args.invalid_pattern_reward,
        "selfplay_round_episodes": args.selfplay_round_episodes,
        "ppo_n_steps": compute_selfplay_rollout_steps(args.episode_len, args.selfplay_round_episodes),
        "cst_design_queries": query_state.n_design_queries,
        "actual_cst_calls": get_cst_call_count(cst),
        "cst_sim_calls": get_cst_call_count(cst),
        "best_surrogate_reward": query_state.best_surrogate_r,
        "global_best_reward": float(vec_env.envs[0].global_best_reward),
        "best_cycle_mean_reward": _json_finite_or_none(callback.best_mean_reward),
        "best_cycle_mean_reward_cycle": callback.best_mean_reward_cycle,
        "cycle_count": len(callback.cycle_history),
        "seed": getattr(args, "seed", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "clip_range": getattr(args, "clip_range", None),
        "ent_coef": getattr(args, "ent_coef", None),
        "tag": getattr(args, "tag", None),
        "timesteps": args.timesteps,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    if cst is not None:
        cst.close()
    print(f"Saved run to {run_dir}")
    augment_v11_run_meta(run_dir, args)
    return run_dir
