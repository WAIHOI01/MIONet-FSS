"""RL design search using trained PPO policy and MIONet world model."""

import json
import os
import tempfile
import time
import zipfile

import numpy as np
import torch
from gymnasium import spaces as gym_spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common.config import DEVICE, N_FREQ, GRID_SIZE, QUARTER_SIZE, STANDARD_FREQS
from inverse.env import (
    V11AngleDispersionEnv,
    V11_CURVE_KEYS,
    V11_CURVE_ANGLES,
    QUARTER_PIXEL_INDICES,
    EPISODE_LEN,
)
from inverse.reward import calculate_sparam_reward_v11, aggregate_stability_reward
from inverse.task import build_active_design_task

# ---------------------------------------------------------------------------
# Legacy curve keys (policies trained with 4 curves instead of 12)
# ---------------------------------------------------------------------------
LEGACY_CURVE_KEYS = ("cur_te_s11", "cur_te_s21", "cur_tm_s11", "cur_tm_s21")


# ===================================================================
#  Helpers
# ===================================================================

def quarter_to_full(quarter):
    """Mirror a (5,5) quarter to a symmetric (10,10) full pattern."""
    q = np.asarray(quarter, dtype=np.float32)
    lo = np.flipud(q)
    le = np.concatenate((q, lo), axis=0)
    ri = np.fliplr(le)
    return np.concatenate((le, ri), axis=1).astype(np.float32)


def parse_curve_key(key):
    """Parse a curve observation key into (pol, param, theta).

    Returns ``None`` if the key format is unrecognised.
    """
    if key in LEGACY_CURVE_KEYS:
        _, pol_name, param = key.split("_")
        return (0 if pol_name == "te" else 1, param, 0.0)
    if key.startswith("cur_") and "_t" in key:
        parts = key[4:]  # strip "cur_"
        if parts.startswith("te_"):
            pol = 0
            param, theta_tag = parts[3:].rsplit("_t", 1)
        elif parts.startswith("tm_"):
            pol = 1
            param, theta_tag = parts[3:].rsplit("_t", 1)
        else:
            return None
        theta = float(theta_tag.replace("p", ".").replace("m", "-"))
        return (pol, param, theta)
    return None


def _select_policy_zip(run_dir):
    """Pick the best available PPO checkpoint from *run_dir*.

    Prefers ``ppo_best_mean_reward.zip``, falls back to ``ppo_final.zip``.
    """
    for filename in ("ppo_best_mean_reward.zip", "ppo_final.zip"):
        candidate = os.path.join(run_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


# ===================================================================
#  Batch prediction helpers (GPU-accelerated)
# ===================================================================

def build_wm_observation_batched(quarter, wm, curve_keys):
    """Batch-predict all S-parameter curves for the observation dict.

    Parameters
    ----------
    quarter : ndarray
        (5, 5) quarter-pattern.
    wm : LocalEnsembleWorldModel
    curve_keys : tuple of str
        Keys to include (e.g. V11_CURVE_KEYS or LEGACY_CURVE_KEYS).

    Returns
    -------
    dict[str, ndarray]
        Keys *curve_keys*, each value ``(N_FREQ,)`` float32 clipped to [0, 1].
    """
    pattern = quarter_to_full(quarter)
    Mt = (
        torch.from_numpy(pattern).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    )

    # Parse all curve requests
    requests = []
    for key in curve_keys:
        parsed = parse_curve_key(key)
        if parsed is not None:
            requests.append((key, parsed))

    if not requests:
        return {k: np.zeros(N_FREQ, dtype=np.float32) for k in curve_keys}

    # Single batch forward pass
    n = len(requests)
    Mt_batch = Mt.repeat(n, 1, 1, 1)
    thetas = torch.tensor([req[1][2] for req in requests], device=DEVICE)
    pols = torch.tensor(
        [req[1][0] for req in requests], device=DEVICE, dtype=torch.long,
    )

    with torch.no_grad():
        s11_batch, s21_batch, _ = wm.predict_stats(
            Mt_batch, theta=thetas, pol=pols,
        )

    obs = {}
    for i, (key, (_pol, param, _theta)) in enumerate(requests):
        values = s11_batch[i] if param == "s11" else s21_batch[i]
        obs[key] = (
            np.clip(np.abs(values.detach().cpu().numpy().flatten()), 0.0, 1.0)
            .astype(np.float32)[:N_FREQ]
        )

    return {
        k: obs.get(k, np.zeros(N_FREQ, dtype=np.float32))[:N_FREQ]
        for k in curve_keys
    }


def evaluate_terminal(pattern_10x10, specs, wm, freqs_ghz):
    """Batch-evaluate the final full pattern against all specs.

    Returns ``(pred_items, total_reward, per_spec_rewards)``.
    """
    pattern = np.asarray(pattern_10x10, dtype=np.float32)
    Mt = (
        torch.from_numpy(pattern).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    )

    n = len(specs)
    Mt_batch = Mt.repeat(n, 1, 1, 1)
    thetas = torch.tensor(
        [float(s["theta"]) for s in specs], device=DEVICE,
    )
    pols = torch.tensor(
        [int(s["pol"]) for s in specs], device=DEVICE, dtype=torch.long,
    )

    with torch.no_grad():
        s11_batch, s21_batch, _ = wm.predict_stats(
            Mt_batch, theta=thetas, pol=pols,
        )

    pred_items = []
    rewards = []
    for i, spec in enumerate(specs):
        s11_np = s11_batch[i].detach().cpu().numpy().flatten()
        s21_np = s21_batch[i].detach().cpu().numpy().flatten()
        reward, _, _ = calculate_sparam_reward_v11(
            np.abs(s11_np), np.abs(s21_np), freqs_ghz, spec,
        )
        rewards.append(float(reward))
        pred_items.append({
            "label": spec["label"],
            "type": spec["type"],
            "pol": int(spec["pol"]),
            "theta": float(spec["theta"]),
            "fc": spec["fc"],
            "bw": spec["bw"],
            "s11": s11_np,
            "s21": s21_np,
        })

    score = aggregate_stability_reward(rewards, specs, stability_weight=0.25)
    return pred_items, float(score), rewards


# ===================================================================
#  Policy curve-key inference
# ===================================================================

def infer_policy_curve_keys(ppo_path):
    """Determine curve keys from a saved PPO policy zip.

    Reads the first feature-extractor weight to check whether the policy
    was trained with 4 legacy curves or 12 v11 curves.

    Parameters
    ----------
    ppo_path : str
        Path to the PPO zip file.

    Returns
    -------
    tuple of str
        Either ``LEGACY_CURVE_KEYS`` or ``V11_CURVE_KEYS``.
    """
    try:
        with zipfile.ZipFile(ppo_path, "r") as archive, \
             tempfile.TemporaryDirectory() as tmp_dir:
            archive.extract("policy.pth", tmp_dir)
            state = torch.load(
                os.path.join(tmp_dir, "policy.pth"), map_location="cpu",
            )
        for key, value in state.items():
            if key.endswith("features_extractor.curve_mlp.0.weight"):
                n_curve = int(value.shape[1]) // int(N_FREQ)
                if n_curve == len(LEGACY_CURVE_KEYS):
                    return LEGACY_CURVE_KEYS
                if n_curve == len(V11_CURVE_KEYS):
                    return V11_CURVE_KEYS
    except Exception as exc:
        print(f"  [warn] Could not infer curve keys: {exc}")
    return V11_CURVE_KEYS


# ===================================================================
#  Model loading
# ===================================================================

def load_rl_model(model_dir, wm, prefer_best_mean=True):
    """Load a PPO policy and create the corresponding environment.

    Parameters
    ----------
    model_dir : str
        Directory containing the PPO zip and optional ``run_meta.json``.
    wm : LocalEnsembleWorldModel
        Pre-loaded world model (shared across calls).
    prefer_best_mean : bool
        If True, try ``ppo_best_mean_reward.zip`` before ``ppo_final.zip``.

    Returns
    -------
    env : V11AngleDispersionEnv
    model : stable_baselines3.PPO
    task_kind : str
        ``"mode"``, ``"angle"``, or ``"joint"``.
    curve_keys : tuple of str
    """
    ppo_path = _select_policy_zip(model_dir)
    if ppo_path is None:
        raise FileNotFoundError(
            f"No PPO zip (ppo_best_mean_reward.zip / ppo_final.zip) "
            f"found in {model_dir}"
        )

    # Read run metadata
    meta_path = os.path.join(model_dir, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        task_kind = meta.get("task_kind", "mode")
        design_mode = meta.get("design_mode", "both")
    else:
        # Heuristic from directory name
        base = os.path.basename(os.path.normpath(model_dir))
        parts = base.split("_")
        task_kind = parts[0] if parts else "mode"
        design_mode = "both"

    # Build environment
    env = V11AngleDispersionEnv(
        wm=wm,
        cst_client=None,
        episode_len=EPISODE_LEN,
        design_mode=design_mode,
        stability_weight=0.25,
        task_kind=task_kind,
        angle_pol="te",
        min_active_quarter_pixels=1,
        invalid_pattern_reward=-10.0,
    )

    # Infer curve keys and adjust observation space to match policy
    curve_keys = infer_policy_curve_keys(ppo_path)
    print(f"  Inferred curve_keys: {len(curve_keys)} curves")

    spaces = env.observation_space.spaces
    # Remove keys the policy does not know about
    for key in V11_CURVE_KEYS:
        if key not in curve_keys and key in spaces:
            del spaces[key]
    # Add legacy keys the policy expects
    for key in LEGACY_CURVE_KEYS:
        if key in curve_keys and key not in spaces:
            spaces[key] = gym_spaces.Box(
                0, 1, (N_FREQ,), dtype=np.float32,
            )

    # Load PPO
    vec_env = DummyVecEnv([lambda e=env: e])
    model = PPO.load(ppo_path, env=vec_env, device=DEVICE)
    print(
        f"  Model loaded: task_kind={task_kind}, "
        f"design_mode={design_mode}, policy={os.path.basename(ppo_path)}"
    )

    return env, model, task_kind, curve_keys


# ===================================================================
#  Episode runner
# ===================================================================

def run_episode(env, model, fc, bw, curve_keys, deterministic=False):
    """Run one greedy design episode.

    Parameters
    ----------
    env : V11AngleDispersionEnv
    model : stable_baselines3.PPO
    fc : float
        Centre frequency in GHz.
    bw : float
        Bandwidth in GHz.
    curve_keys : tuple of str
    deterministic : bool
        If True, use ``model.predict(deterministic=True)``.

    Returns
    -------
    pattern_10x10 : ndarray  (10, 10)
    pred_items : list of dict
    total_reward : float
    """
    # Build task specs
    specs, masks = build_active_design_task(
        fc, bw, env.task_kind, "transmit",
        angle_pol=getattr(env, "angle_pol", "te"),
    )

    # Override env target so its internal state matches
    env._override_target = (fc, bw, "transmit")
    env._assign_task()

    quarter = np.zeros((QUARTER_SIZE, QUARTER_SIZE), dtype=np.float32)
    n_steps = min(EPISODE_LEN, len(QUARTER_PIXEL_INDICES))

    for idx in range(n_steps):
        # Build observation dict
        obs = {
            "image": quarter.astype(np.float32)[np.newaxis, :, :].copy(),
            "te_target_s11": masks["te_s11"].astype(np.float32).copy(),
            "te_target_s21": masks["te_s21"].astype(np.float32).copy(),
            "tm_target_s11": masks["tm_s11"].astype(np.float32).copy(),
            "tm_target_s21": masks["tm_s21"].astype(np.float32).copy(),
        }
        obs.update(build_wm_observation_batched(quarter, env.wm, curve_keys))

        action, _ = model.predict(obs, deterministic=deterministic)
        i, j = QUARTER_PIXEL_INDICES[idx]
        quarter[i, j] = float(int(np.asarray(action).reshape(-1)[0]))

    pattern_10x10 = quarter_to_full(quarter)
    pred_items, reward, _ = evaluate_terminal(
        pattern_10x10, specs, env.wm, STANDARD_FREQS,
    )
    return pattern_10x10, pred_items, reward


# ===================================================================
#  Search
# ===================================================================

def rl_search(env, model, fc, bw, curve_keys, n_episodes=600,
              deterministic=False, verbose=True):
    """Search for the best design via RL rollouts.

    Parameters
    ----------
    env : V11AngleDispersionEnv
    model : stable_baselines3.PPO
    fc : float
    bw : float
    curve_keys : tuple of str
    n_episodes : int
        Number of greedy episodes to run.
    deterministic : bool
        Passed through to ``model.predict``.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: ``pattern``, ``pred_items``, ``best_reward``,
        ``best_episode``, ``mean_reward``, ``std_reward``,
        ``all_rewards``, ``n_episodes``, ``elapsed``.
    """
    best_pattern = None
    best_pred_items = None
    best_reward = -float("inf")
    best_ep = -1
    all_rewards = []

    t_start = time.time()
    for ep in range(n_episodes):
        pattern, pred_items, reward = run_episode(
            env, model, fc, bw, curve_keys, deterministic=deterministic,
        )
        all_rewards.append(reward)
        if reward > best_reward:
            best_reward = reward
            best_pattern = pattern.copy()
            best_pred_items = [
                {k: v for k, v in pi.items()} for pi in pred_items
            ]
            best_ep = ep

        if verbose and (ep + 1) % 50 == 0:
            mean_r = float(np.mean(all_rewards[-50:]))
            print(
                f"    RL ep {ep + 1:>4d}/{n_episodes} | "
                f"best={best_reward:.4f}  recent_mean={mean_r:.4f}"
            )

    elapsed = time.time() - t_start
    mean_r = float(np.mean(all_rewards))
    std_r = float(np.std(all_rewards))

    if verbose:
        print(
            f"    Done: best={best_reward:.4f} (ep {best_ep}), "
            f"mean={mean_r:.4f}+/-{std_r:.4f}, time={elapsed:.1f}s"
        )

    return {
        "pattern": best_pattern,
        "pred_items": best_pred_items,
        "best_reward": float(best_reward),
        "best_episode": best_ep,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "all_rewards": all_rewards,
        "n_episodes": n_episodes,
        "elapsed": elapsed,
    }
