"""Dual-mode physics-aware loss with energy conservation penalty."""
import torch
import torch.nn as nn


class DualModePhysicsLoss(nn.Module):
    """MSE loss + energy conservation penalty for TE+TM dual-mode prediction.

    loss = sum_i MSE(pred_i, target_i)
           + phys_weight * sum_{pol in {TE,TM}} mean((|S11|^2 + |S21|^2 - 1)^2)
    """
    def __init__(self, phys_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.phys_weight = phys_weight

    def forward(self, p_te_s11, p_te_s21, p_tm_s11, p_tm_s21,
                t_te_s11, t_te_s21, t_tm_s11, t_tm_s21):
        loss_data = (
            self.mse(p_te_s11, t_te_s11) + self.mse(p_te_s21, t_te_s21) +
            self.mse(p_tm_s11, t_tm_s11) + self.mse(p_tm_s21, t_tm_s21)
        )
        energy_te = p_te_s11 ** 2 + p_te_s21 ** 2
        energy_tm = p_tm_s11 ** 2 + p_tm_s21 ** 2
        loss_phys = (
            torch.mean((energy_te - 1.0) ** 2) +
            torch.mean((energy_tm - 1.0) ** 2)
        )
        return loss_data + self.phys_weight * loss_phys
