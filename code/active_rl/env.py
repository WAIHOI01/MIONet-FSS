"""RL environment: multi-angle TE/TM FSS quarter-pixel placement with
uncertainty-gated end-of-episode CST verification.

The single V11AngleDispersionEnv merges the former v9 AngleDispersionEnv body
with the v11 subclass (12 S-parameter curve observations + fixed-grid task
sampling). Gate/query functions are reached via the cst_gate module so the
tau monitor can wrap them.
"""

import random

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
import torch

from active_rl import cst_gate
from active_rl.config import (
    DEVICE,
    GRID_SIZE,
    QUARTER_SIZE,
    QUARTER_PIXEL_INDICES,
    N_FREQ,
    STANDARD_FREQS,
    DEFAULT_STABILITY_WEIGHT,
    DEFAULT_MIN_ACTIVE_QUARTER_PIXELS,
    DEFAULT_INVALID_PATTERN_REWARD,
)
from active_rl.reward import aggregate_stability_reward, _mag01
from active_rl import tasks
from active_rl.tasks import (
    build_active_design_task,
    sample_active_design_task,
    normalize_pol,
    V11_CURVE_ANGLES,
    V11_CURVE_KEYS,
    v11_curve_key,
)


def reconstruct_full_pattern(quarter):
    quarter = np.asarray(quarter, dtype=np.float32)
    lower = np.flipud(quarter)
    left = np.concatenate((quarter, lower), axis=0)
    right = np.fliplr(left)
    return np.concatenate((left, right), axis=1).astype(np.float32)


def evaluate_pattern_validity(quarter, min_active_quarter_pixels=DEFAULT_MIN_ACTIVE_QUARTER_PIXELS,
                              invalid_reward=DEFAULT_INVALID_PATTERN_REWARD):
    quarter = np.asarray(quarter, dtype=np.float32)
    active = int(np.count_nonzero(quarter > 0.5))
    min_active = max(0, int(min_active_quarter_pixels))
    valid = active >= min_active
    return {
        "valid": bool(valid),
        "active_quarter_pixels": active,
        "min_active_quarter_pixels": min_active,
        "reward": None if valid else float(invalid_reward),
    }
def summarize_uncertainty(wm, pattern_10x10, specs):
    if hasattr(wm, "estimate_uncertainty"):
        return float(wm.estimate_uncertainty(pattern_10x10, specs))
    return 1.0


class V11AngleDispersionEnv(gym.Env):
    """Multi-angle TE/TM joint FSS design with shared fc/bw and opposite functions.

    CST triggered only at episode end; reward = CST reward when available.
    Adds the v11 fixed-target grid sampling and 12 predicted S-parameter
    curve observations (3 angles x 2 pols x {s11, s21}) to the state.
    """

    metadata = {"render_modes": []}

    def __init__(self, wm, cst_client=None, episode_len=25, design_mode="both",
                 stability_weight=DEFAULT_STABILITY_WEIGHT, active_query_state=None,
                 cst_query_mode="representative", task_kind="joint", angle_pol="te",
                 min_active_quarter_pixels=DEFAULT_MIN_ACTIVE_QUARTER_PIXELS,
                 invalid_pattern_reward=DEFAULT_INVALID_PATTERN_REWARD,
                 target_fc_noise=None, target_bw_noise=None):
        super().__init__()
        self.wm = wm
        self.cst = cst_client
        self.query_state = active_query_state or cst_gate.ActiveQueryState()
        self.cst_query_mode = cst_query_mode
        self.task_kind = str(task_kind)
        self.angle_pol = normalize_pol(angle_pol)
        self.episode_len = min(int(episode_len), len(QUARTER_PIXEL_INDICES))
        self.design_mode = design_mode
        self.stability_weight = float(stability_weight)
        self.min_active_quarter_pixels = int(min_active_quarter_pixels)
        self.invalid_pattern_reward = float(invalid_pattern_reward)
        self.freqs_ghz = STANDARD_FREQS
        self._override_target = None
        self.target_fc_noise = (
            tasks.TARGET_FC_NOISE if target_fc_noise is None else float(target_fc_noise))
        self.target_bw_noise = (
            tasks.TARGET_BW_NOISE if target_bw_noise is None else float(target_bw_noise))

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 1, (1, QUARTER_SIZE, QUARTER_SIZE), dtype=np.float32),
            "te_target_s11": spaces.Box(0, 1, (N_FREQ,), dtype=np.float32),
            "te_target_s21": spaces.Box(0, 1, (N_FREQ,), dtype=np.float32),
            "tm_target_s11": spaces.Box(0, 1, (N_FREQ,), dtype=np.float32),
            "tm_target_s21": spaces.Box(0, 1, (N_FREQ,), dtype=np.float32),
        })
        for key in V11_CURVE_KEYS:
            self.observation_space.spaces[key] = spaces.Box(
                0, 1, (N_FREQ,), dtype=np.float32)

        self.current_quarter = None
        self.current_M = None
        self.step_count = 0
        self.best_reward = -float("inf")
        self.best_M = None
        self.best_quarter = None
        self.global_best_reward = -float("inf")
        self.global_best_M = None
        self.global_best_quarter = None
        self.global_best_specs = None
        self.global_best_fc_0 = None
        self.global_best_bw_0 = None
        self.global_best_te_type = None
        self.prev_func_r = None
        self.best_surrogate_r = self.query_state.best_surrogate_r
        self.cst_triggered = False
        self.last_uncertainty = 1.0

        self.current_specs = []
        self.current_fc_0 = None
        self.current_bw_0 = None
        self.current_te_type = None
        self.current_masks = {
            "te_s11": np.zeros(N_FREQ, dtype=np.float32),
            "te_s21": np.zeros(N_FREQ, dtype=np.float32),
            "tm_s11": np.zeros(N_FREQ, dtype=np.float32),
            "tm_s21": np.zeros(N_FREQ, dtype=np.float32),
        }
        self.last_func_r = -float("inf")
        self.episode_index = 0

    def set_target(self, fc_0, bw_0, te_design_type="transmit"):
        self._override_target = (float(fc_0), float(bw_0), str(te_design_type))

    def clear_target_override(self):
        self._override_target = None

    def _assign_task(self):
        if self._override_target is not None:
            fc_0, bw_0, te_design_type = self._override_target
            specs, masks = build_active_design_task(
                fc_0, bw_0, self.task_kind, te_design_type,
                angle_pol=self.angle_pol)
        else:
            specs, masks, fc_0, bw_0, te_design_type = sample_active_design_task(
                self.freqs_ghz, task_kind=self.task_kind,
                design_mode=self.design_mode, angle_pol=self.angle_pol,
                fc_noise=self.target_fc_noise, bw_noise=self.target_bw_noise)
        self.current_specs = specs
        self.current_masks = masks
        self.current_fc_0 = float(fc_0)
        self.current_bw_0 = float(bw_0)
        self.current_te_type = str(te_design_type)

    def _current_full_pattern(self):
        return reconstruct_full_pattern(self.current_quarter.detach().cpu().numpy())

    def _evaluate_current_pattern(self):
        full = self._current_full_pattern()
        full_t = torch.from_numpy(full).float().to(DEVICE)
        rewards = [float(self.wm.evaluate(full_t, spec)) for spec in self.current_specs]
        score = aggregate_stability_reward(
            rewards, self.current_specs, stability_weight=self.stability_weight)
        return full_t, score

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.current_quarter = torch.zeros(QUARTER_SIZE, QUARTER_SIZE, device=DEVICE)
        self.current_M = torch.zeros(GRID_SIZE, GRID_SIZE, device=DEVICE)
        self.step_count = 0
        self.episode_index += 1
        self.best_reward = -float("inf")
        self.best_M = None
        self.best_quarter = None
        self.best_surrogate_r = self.query_state.best_surrogate_r
        self.cst_triggered = False
        self.last_uncertainty = 1.0
        self._assign_task()

        self.current_M, self.prev_func_r = self._evaluate_current_pattern()
        self.last_func_r = float(self.prev_func_r)
        return self._obs(), {}

    def _current_sparam_observation(self):
        zeros = np.zeros(N_FREQ, dtype=np.float32)
        curves = {key: zeros.copy() for key in V11_CURVE_KEYS}
        if self.current_M is None:
            return curves

        full = self.current_M
        if isinstance(full, torch.Tensor):
            full_t = full.detach().float().to(DEVICE)
        else:
            full_t = torch.from_numpy(np.asarray(full, dtype=np.float32)).float().to(DEVICE)
        geom = full_t.unsqueeze(0).unsqueeze(0)

        try:
            with torch.no_grad():
                for theta in V11_CURVE_ANGLES:
                    for pol in (0, 1):
                        s11, s21 = self.wm.predict(geom, theta=float(theta), pol=int(pol))
                        curves[v11_curve_key(pol, "s11", theta)] = _mag01(
                            s11.detach().cpu().numpy().flatten())[:N_FREQ]
                        curves[v11_curve_key(pol, "s21", theta)] = _mag01(
                            s21.detach().cpu().numpy().flatten())[:N_FREQ]
        except Exception:
            return curves

        return curves

    def _obs(self):
        obs = {
            "image": self.current_quarter.detach().cpu().numpy()[np.newaxis, :, :].copy(),
            "te_target_s11": self.current_masks["te_s11"].copy(),
            "te_target_s21": self.current_masks["te_s21"].copy(),
            "tm_target_s11": self.current_masks["tm_s11"].copy(),
            "tm_target_s21": self.current_masks["tm_s21"].copy(),
        }
        obs.update(self._current_sparam_observation())
        return obs

    def step(self, action):
        idx = self.step_count
        i, j = QUARTER_PIXEL_INDICES[idx]
        self.current_quarter[i, j] = float(int(action))
        self.step_count += 1

        self.current_M, surrogate_func_r = self._evaluate_current_pattern()
        pattern_10x10 = self.current_M.detach().cpu().numpy().astype(np.float32)
        uncertainty = summarize_uncertainty(self.wm, pattern_10x10, self.current_specs)
        self.last_uncertainty = uncertainty

        trigger_cst = False
        cst_info = {}
        trigger_reason = {"predicted_new_best": False, "high_uncertainty": False, "forced": False}
        design_complete = self.step_count >= self.episode_len

        final_func_r = surrogate_func_r
        reward = 0.0
        validity_info = evaluate_pattern_validity(
            self.current_quarter.detach().cpu().numpy(),
            min_active_quarter_pixels=self.min_active_quarter_pixels,
            invalid_reward=self.invalid_pattern_reward,
        )

        if design_complete and not validity_info["valid"]:
            final_func_r = float(validity_info["reward"])
            cst_info = {
                "surrogate_func_r": float(surrogate_func_r),
                "invalid_pattern": True,
                "invalid_reason": "too_few_active_quarter_pixels",
                "active_quarter_pixels": int(validity_info["active_quarter_pixels"]),
                "min_active_quarter_pixels": int(validity_info["min_active_quarter_pixels"]),
                "cst_calls": cst_gate.get_cst_call_count(self.cst),
                "cst_design_queries": self.query_state.n_design_queries,
            }
        elif design_complete and self.cst is not None and not self.cst_triggered:
            trigger_cst, trigger_reason = cst_gate.should_trigger_cst(
                query_state=self.query_state,
                pattern_10x10=pattern_10x10,
                current_func_r=surrogate_func_r,
                uncertainty=uncertainty,
                episode_index=self.episode_index,
            )
            if trigger_cst:
                self.cst_triggered = True

                query_specs = cst_gate.select_cst_query_specs(self.current_specs, self.cst_query_mode)
                cst_rewards, cst_specs, cst_success = cst_gate.evaluate_cst_query_specs(
                    self.cst, self.wm, pattern_10x10, query_specs, self.freqs_ghz)

                if cst_success > 0:
                    cst_r = aggregate_stability_reward(cst_rewards, cst_specs, self.stability_weight)
                    self.query_state.mark_queried(pattern_10x10, surrogate_func_r)
                    self.best_surrogate_r = self.query_state.best_surrogate_r
                    self.wm.finetune(steps=10, batch_size=16)
                    final_func_r = cst_r
                    cst_info = {
                        "cst_func_r": float(cst_r),
                        "surrogate_func_r": float(surrogate_func_r),
                        "cst_calls": cst_gate.get_cst_call_count(self.cst),
                        "cst_success": cst_success,
                        "cst_query_mode": self.cst_query_mode,
                        "cst_design_queries": self.query_state.n_design_queries,
                    }
                else:
                    self.query_state.observe_surrogate(surrogate_func_r)
                    self.best_surrogate_r = self.query_state.best_surrogate_r
            else:
                self.query_state.observe_surrogate(surrogate_func_r)
                self.best_surrogate_r = self.query_state.best_surrogate_r
                cst_info = {
                    "surrogate_func_r": float(surrogate_func_r),
                    "cst_calls": cst_gate.get_cst_call_count(self.cst),
                    "cst_design_queries": self.query_state.n_design_queries,
                }
        elif design_complete:
            self.query_state.observe_surrogate(surrogate_func_r)
            self.best_surrogate_r = self.query_state.best_surrogate_r

        if design_complete:
            reward = float(final_func_r)
        self.prev_func_r = final_func_r
        self.last_func_r = float(final_func_r)

        if design_complete:
            self.best_reward = final_func_r
            self.best_M = self.current_M.clone()
            self.best_quarter = self.current_quarter.clone()
            if final_func_r > self.global_best_reward:
                self.global_best_reward = final_func_r
                self.global_best_M = self.current_M.clone()
                self.global_best_quarter = self.current_quarter.clone()
                self.global_best_specs = [dict(spec) for spec in self.current_specs]
                self.global_best_fc_0 = float(self.current_fc_0)
                self.global_best_bw_0 = float(self.current_bw_0)
                self.global_best_te_type = str(self.current_te_type)

        terminated = False
        truncated = self.step_count >= self.episode_len
        info = {
            "best_reward": self.best_reward,
            "trigger_cst": trigger_cst,
            "fc_0": self.current_fc_0,
            "bw_0": self.current_bw_0,
            "te_type": self.current_te_type,
            "uncertainty": float(uncertainty),
            "trigger_reason": trigger_reason,
            "episode_index": self.episode_index,
            "design_complete": bool(design_complete),
            "active_quarter_pixels": int(validity_info["active_quarter_pixels"]),
            "min_active_quarter_pixels": int(validity_info["min_active_quarter_pixels"]),
            "invalid_pattern": bool(not validity_info["valid"]),
            **cst_info,
        }
        if design_complete:
            info.update({
                "episode_reward": float(final_func_r),
                "terminal_pattern_10x10": pattern_10x10.copy(),
                "terminal_quarter_5x5": self.current_quarter.detach().cpu().numpy().astype(np.float32).copy(),
                "terminal_specs": [dict(spec) for spec in self.current_specs],
                "terminal_fc_0": float(self.current_fc_0),
                "terminal_bw_0": float(self.current_bw_0),
                "terminal_te_type": str(self.current_te_type),
            })
        return self._obs(), reward, terminated, truncated, info
