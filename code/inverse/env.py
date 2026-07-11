"""V11 Angle-Dispersion RL Environment for FSS inverse design.

Self-contained --- no v9/v8 inheritance.  Uses the MIONet world model for
S-parameter prediction during the sequential quarter-pixel placement process.
"""

import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from common.config import DEVICE, N_FREQ, GRID_SIZE, QUARTER_SIZE, STANDARD_FREQS
from inverse.reward import (
    V11_GUARD_GHZ,
    V11_PASS_SCORE_TARGET,
    V11_OUT_PENALTY_TARGET,
    DEFAULT_STABILITY_WEIGHT,
    calculate_sparam_reward_v11,
    aggregate_stability_reward,
)
from inverse.task import build_active_design_task

# ---------------------------------------------------------------------------
# Constants  (inlined from v9 / v11 so this file has no v9/v8 imports)
# ---------------------------------------------------------------------------

V11_CURVE_ANGLES = (0.0, 30.0, 60.0)          # incident angles for S-param curves
V11_FC_LIST       = [8.0, 10.0, 12.0, 14.0]   # centre-frequency grid (GHz)
V11_BW_LIST       = [1.0,  2.0,  3.0,  4.0]   # bandwidth grid (GHz)
V11_TARGET_FC_NOISE  = 0.0
V11_TARGET_BW_NOISE  = 0.0
FEASIBLE_FC_RANGE = (0.1, 20.0)
FEASIBLE_BW_RANGE = (0.1, 19.9)

SWEEP_DESIGN_TYPES   = ("transmit", "reflect")

EPISODE_LEN           = 25
INVALID_PATTERN_REWARD = -10.0
MIN_ACTIVE_QUARTER_PIXELS = 1

# 5x5 quarter-pixel indices (row-major)
QUARTER_PIXEL_INDICES = [
    (r, c) for r in range(QUARTER_SIZE) for c in range(QUARTER_SIZE)
]

# Target-mask keys (same shape as v9)
V11_TARGET_KEYS = (
    "te_target_s11", "te_target_s21", "tm_target_s11", "tm_target_s21",
)


def _v11_angle_tag(theta: float) -> str:
    """Convert a float angle to the v11 observation-key suffix.

    Uses the same ``{float(theta):g}`` format as v11 so that keys are
    consistent with any PPO policy trained under v11.
    """
    return f"{float(theta):g}".replace(".", "p").replace("-", "m")


def _v11_curve_key(pol: int, param: str, theta: float) -> str:
    """Build a single S-parameter curve key, e.g. ``cur_te_s11_t0``."""
    pol_name = "te" if int(pol) == 0 else "tm"
    return f"cur_{pol_name}_{str(param).lower()}_t{_v11_angle_tag(theta)}"


V11_CURVE_KEYS = tuple(
    _v11_curve_key(pol, param, theta)
    for theta in V11_CURVE_ANGLES
    for pol in (0, 1)
    for param in ("s11", "s21")
)
"""12 keys: 3 angles x 2 polarisations x 2 parameters (s11, s21)."""


# ===================================================================
#  Environment
# ===================================================================

class V11AngleDispersionEnv(gym.Env):
    """V11 RL environment --- sequential quarter-pattern FSS design.

    State
    -----
    - ``image``:  5x5 quarter grid  ``(1, 5, 5)``
    - 4 target-mask vectors  ``(N_FREQ,)``  each
    - 12 predicted S-parameter curve vectors  ``(N_FREQ,)``  each

    Action
    ------
    Discrete(25) --- pick one 5x5 quarter pixel to set to 1 (metal).

    Episode
    -------
    Up to 25 steps.  Reward is only given at the terminal step (v11
    in-band-score minus out-band-penalty, aggregated across all specs
    with a stability penalty).
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        wm,
        cst_client=None,
        episode_len: int = EPISODE_LEN,
        design_mode: str = "transmit",
        stability_weight: float = DEFAULT_STABILITY_WEIGHT,
        task_kind: str = "mode",
        angle_pol: str = "te",
        min_active_quarter_pixels: int = MIN_ACTIVE_QUARTER_PIXELS,
        invalid_pattern_reward: float = INVALID_PATTERN_REWARD,
        target_fc_noise: float | None = None,
        target_bw_noise: float | None = None,
    ):
        super().__init__()

        # --- external clients -----------------------------------------
        self.wm        = wm
        self.cst_client = cst_client

        # --- hyper-parameters -----------------------------------------
        self.episode_len             = int(episode_len)
        self.design_mode             = str(design_mode)
        self.stability_weight        = float(stability_weight)
        self.task_kind               = str(task_kind)
        self.angle_pol               = str(angle_pol)
        self.min_active_quarter_pixels = int(min_active_quarter_pixels)
        self.invalid_pattern_reward  = float(invalid_pattern_reward)
        self.target_fc_noise = (
            float(target_fc_noise) if target_fc_noise is not None else V11_TARGET_FC_NOISE
        )
        self.target_bw_noise = (
            float(target_bw_noise) if target_bw_noise is not None else V11_TARGET_BW_NOISE
        )

        # --- observation space (inlined from v9 + v11 curve keys) -----
        obs_spaces: dict[str, spaces.Box] = {
            "image": spaces.Box(
                0, 1, (1, QUARTER_SIZE, QUARTER_SIZE), dtype=np.float32,
            ),
        }
        for key in V11_TARGET_KEYS:
            obs_spaces[key] = spaces.Box(0, 1, (N_FREQ,), dtype=np.float32)
        for key in V11_CURVE_KEYS:
            obs_spaces[key] = spaces.Box(0, 1, (N_FREQ,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

        # --- action space (pick one quarter pixel) --------------------
        self.action_space  = spaces.Discrete(len(QUARTER_PIXEL_INDICES))
        self.pixel_indices = list(QUARTER_PIXEL_INDICES)

        # --- internal episode state -----------------------------------
        self.current_step    = 0
        self.current_quarter: np.ndarray | None = None   # (5, 5) float32
        self.current_M:      np.ndarray | None = None   # (10,10) float32
        self.current_specs:  list | None       = None
        self.current_masks:  dict | None       = None
        self._override_target: tuple | None    = None

        # best-so-far tracking
        self.best_reward  = -float("inf")
        self.best_pattern: np.ndarray | None = None

    # ------------------------------------------------------------------
    #  Mirroring helper  (inlined from v9 `reconstruct_full_pattern`)
    # ------------------------------------------------------------------

    @staticmethod
    def _quarter_to_full(quarter: np.ndarray) -> np.ndarray:
        """Mirror a ``(5,5)`` quarter to a symmetric ``(10,10)`` full pattern."""
        q  = np.asarray(quarter, dtype=np.float32)
        lo = np.flipud(q)
        le = np.concatenate((q, lo), axis=0)
        ri = np.fliplr(le)
        return np.concatenate((le, ri), axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    #  Task assignment  (inlined from v11 `_assign_task`)
    # ------------------------------------------------------------------

    def _assign_task(self) -> None:
        """Sample (or read from override) the target *(fc, bw, design_type)*."""
        if self._override_target is not None:
            fc_0, bw_0, te_design_type = self._override_target
            specs, masks = build_active_design_task(
                fc_0, bw_0, self.task_kind, te_design_type, angle_pol=self.angle_pol,
            )
        else:
            # --- random sampling from the v11 fixed grid --------------
            fc_0 = float(random.choice(V11_FC_LIST))
            bw_0 = float(random.choice(V11_BW_LIST))

            # apply optional noise then clip
            if self.target_fc_noise > 0.0:
                fc_0 += random.uniform(-self.target_fc_noise, self.target_fc_noise)
            if self.target_bw_noise > 0.0:
                bw_0 += random.uniform(-self.target_bw_noise, self.target_bw_noise)
            fc_0 = float(np.clip(fc_0, *FEASIBLE_FC_RANGE))
            bw_0 = float(np.clip(bw_0, *FEASIBLE_BW_RANGE))

            # choose TE design type respecting design_mode
            if self.design_mode == "both":
                te_design_type = random.choice(SWEEP_DESIGN_TYPES)
            elif self.design_mode in SWEEP_DESIGN_TYPES:
                te_design_type = self.design_mode
            else:
                raise ValueError(f"Unknown design_mode: {self.design_mode}")

            specs, masks = build_active_design_task(
                fc_0, bw_0, self.task_kind, te_design_type, angle_pol=self.angle_pol,
            )

        self.current_specs = specs
        self.current_masks = masks

    # ------------------------------------------------------------------
    #  S-parameter observation  (inlined from v11)
    # ------------------------------------------------------------------

    def _current_sparam_observation(self) -> dict[str, np.ndarray]:
        """Batch-predict all S-param curves for the current quarter pattern.

        Returns a dict keyed by ``V11_CURVE_KEYS``; each value is an
        ``(N_FREQ,)`` float32 array clipped to [0, 1].
        """
        zeros = np.zeros(N_FREQ, dtype=np.float32)
        curves: dict[str, np.ndarray] = {key: zeros.copy() for key in V11_CURVE_KEYS}

        if self.current_M is None:
            return curves

        full_t = torch.from_numpy(
            np.asarray(self.current_M, dtype=np.float32)
        ).float().to(DEVICE)
        geom = full_t.unsqueeze(0).unsqueeze(0)   # (1, 1, 10, 10)

        try:
            with torch.no_grad():
                for theta in V11_CURVE_ANGLES:
                    for pol in (0, 1):
                        s11, s21 = self.wm.predict(
                            geom, theta=float(theta), pol=int(pol),
                        )
                        s11_np = s11.detach().cpu().numpy().flatten()[:N_FREQ]
                        s21_np = s21.detach().cpu().numpy().flatten()[:N_FREQ]

                        key_s11 = _v11_curve_key(pol, "s11", theta)
                        key_s21 = _v11_curve_key(pol, "s21", theta)
                        curves[key_s11] = np.clip(
                            np.abs(s11_np), 0.0, 1.0,
                        ).astype(np.float32)
                        curves[key_s21] = np.clip(
                            np.abs(s21_np), 0.0, 1.0,
                        ).astype(np.float32)
        except Exception:
            pass

        return curves

    # ------------------------------------------------------------------
    #  Observation builder  (inlined from v9 `_obs` + v11 extension)
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Assemble the full observation dict for PPO."""
        obs: dict[str, np.ndarray] = {
            "image": self.current_quarter.astype(np.float32)[np.newaxis, :, :].copy(),
        }

        # target masks
        if self.current_masks is not None:
            obs["te_target_s11"] = self.current_masks["te_s11"].astype(np.float32).copy()
            obs["te_target_s21"] = self.current_masks["te_s21"].astype(np.float32).copy()
            obs["tm_target_s11"] = self.current_masks["tm_s11"].astype(np.float32).copy()
            obs["tm_target_s21"] = self.current_masks["tm_s21"].astype(np.float32).copy()
        else:
            for key in V11_TARGET_KEYS:
                obs[key] = np.zeros(N_FREQ, dtype=np.float32)

        # v11 S-parameter curves
        obs.update(self._current_sparam_observation())
        return obs

    # ------------------------------------------------------------------
    #  Reward  (terminal-only v11 reward)
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """Compute the v11 terminal reward across all specs."""
        if self.current_M is None:
            return self.invalid_pattern_reward

        n_metal = int(np.sum(self.current_quarter))
        if n_metal < self.min_active_quarter_pixels:
            return self.invalid_pattern_reward

        full_t = torch.from_numpy(
            np.asarray(self.current_M, dtype=np.float32),
        ).float().to(DEVICE)
        geom = full_t.unsqueeze(0).unsqueeze(0)

        rewards: list[float] = []
        for spec in self.current_specs:
            theta = float(spec["theta"])
            pol   = int(spec["pol"])
            try:
                with torch.no_grad():
                    s11, s21 = self.wm.predict(geom, theta=theta, pol=pol)
                s11_np = s11.detach().cpu().numpy().flatten()
                s21_np = s21.detach().cpu().numpy().flatten()
                r, _, _ = calculate_sparam_reward_v11(
                    s11_np, s21_np, STANDARD_FREQS, spec,
                )
                rewards.append(float(r))
            except Exception:
                rewards.append(0.0)

        if not rewards:
            return self.invalid_pattern_reward

        score = aggregate_stability_reward(
            rewards, self.current_specs, stability_weight=self.stability_weight,
        )
        return float(score)

    # ------------------------------------------------------------------
    #  Gym API
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Start a new episode with a freshly sampled task."""
        super().reset(seed=seed)
        self.current_step    = 0
        self.current_quarter = np.zeros(
            (QUARTER_SIZE, QUARTER_SIZE), dtype=np.float32,
        )
        self.current_M = None
        self._assign_task()
        return self._build_observation(), {}

    def step(
        self, action: int,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Apply one quarter-pixel placement and advance the episode."""
        action = int(action)
        r, c = self.pixel_indices[action]
        self.current_quarter[r, c] = 1.0
        self.current_M = self._quarter_to_full(self.current_quarter)
        self.current_step += 1

        done = (
            self.current_step >= self.episode_len
            or np.sum(self.current_quarter) >= len(self.pixel_indices)
        )
        truncated = False

        reward = self._compute_reward() if done else 0.0
        obs    = self._build_observation()

        # update best-so-far
        if done and reward > self.best_reward:
            self.best_reward  = reward
            self.best_pattern = self.current_M.copy()

        return obs, float(reward), done, truncated, {}

    # ------------------------------------------------------------------
    #  Target override  (for sweep / evaluation)
    # ------------------------------------------------------------------

    def set_target(self, fc: float, bw: float, te_design_type: str = "transmit") -> None:
        """Fix the target for the next ``reset()`` call."""
        self._override_target = (float(fc), float(bw), str(te_design_type))

    def clear_target_override(self) -> None:
        """Resume random task sampling."""
        self._override_target = None
