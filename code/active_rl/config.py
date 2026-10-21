"""Shared constants and paths for the active_rl package.

Imported first by every module: the KMP_DUPLICATE_LIB_OK workaround and the
warnings filter are set before any torch import happens.
"""

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 10
N_FREQ = 1001
ENSEMBLE_SIZE = 3

QUARTER_SIZE = GRID_SIZE // 2
QUARTER_PIXEL_INDICES = [(i, j) for i in range(QUARTER_SIZE) for j in range(QUARTER_SIZE)]
STANDARD_FREQS = np.linspace(0.01, 20.0, N_FREQ, dtype=np.float32)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_world_model")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- joint angle-stability config ----
ANGLES_DEG = [0, 10, 20, 30, 40, 50, 60]
DEFAULT_STABILITY_WEIGHT = 0.25
FEASIBLE_FC_RANGE = (8.0, 14.0)
FEASIBLE_BW_RANGE = (1.0, 4.0)

SWEEP_FC_LIST = [8.0, 10.0, 12.0, 14.0]
SWEEP_BW_LIST = [1.0, 2.0, 3.0, 4.0]
SWEEP_DESIGN_TYPES = ["transmit", "reflect"]
UNCERTAINTY_THRESHOLD = 0.18
SURROGATE_IMPROVE_MARGIN = 0.05
EPSILON_CST_FORCE = 0.0
DEFAULT_CST_BUDGET = 0
DEFAULT_CST_INTERVAL_EPISODES = 10
DEFAULT_MIN_SURROGATE_REWARD = -1.5
DEFAULT_MIN_ACTIVE_QUARTER_PIXELS = 1
DEFAULT_INVALID_PATTERN_REWARD = -10.0
M1_MODEL_PATH = os.path.join(SCRIPT_DIR, "server", "best_system_eq_robust_dropout_M1_c_theta0p0to60p0_order16.pth")
M2_MODEL_PATH = os.path.join(SCRIPT_DIR, "server", "best_system_eq_robust_dropout_M2_c_theta0p0to60p0_order16.pth")

# ---- v11 fixed-target constants ----
V11_PASS_SCORE_TARGET = 0.89
V11_OUT_PENALTY_TARGET = 0.31
V11_GUARD_GHZ = 0.25
