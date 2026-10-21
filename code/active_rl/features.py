"""PPO feature extractor: image CNN + target-mask MLP + S-parameter curve MLP."""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from active_rl.config import N_FREQ
from active_rl.tasks import V11_TARGET_KEYS, V11_CURVE_KEYS


class V11AngleDispersionFeatures(BaseFeaturesExtractor):
    def __init__(self, obs_space, dim=256):
        super().__init__(obs_space, dim)
        n_img = obs_space.spaces["image"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_img, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        n_target = N_FREQ * len(V11_TARGET_KEYS)
        n_curve = N_FREQ * len(V11_CURVE_KEYS)
        self.target_mlp = nn.Sequential(
            nn.Linear(n_target, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.curve_mlp = nn.Sequential(
            nn.Linear(n_curve, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),
        )
        self.fuse = nn.Sequential(nn.Linear(128 + 128 + 128, dim), nn.ReLU())

    def forward(self, obs):
        targets = torch.cat([obs[key] for key in V11_TARGET_KEYS], dim=1)
        curves = torch.cat([obs[key] for key in V11_CURVE_KEYS], dim=1)
        return self.fuse(torch.cat([
            self.cnn(obs["image"]),
            self.target_mlp(targets),
            self.curve_mlp(curves),
        ], dim=1))
