"""Global constants for MIONet-FSS."""
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frequency grid
N_FREQ = 1001
STANDARD_FREQS = np.linspace(0.01, 20.0, N_FREQ, dtype=np.float32)

# Grid
GRID_SIZE = 10
QUARTER_SIZE = GRID_SIZE // 2  # 5

# Episode
EPISODE_LEN = 25

# Default paths (relative to PROJECT_ROOT which is MIONet-FSS/)
DATASET_PATH = os.path.join(PROJECT_ROOT, "..", "code", "dataset_forward_mode_angle_v2.pt")
M1_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "..", "code", "server",
    "best_system_eq_robust_dropout_M1_c_theta0p0to60p0_order16.pth"
)
