"""Local ensemble world model: 3 MIONet copies with progressive Gaussian noise
for variance-based uncertainty estimates and online finetuning from CST data.
"""

import random
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from active_rl.config import (
    DEVICE,
    N_FREQ,
    ENSEMBLE_SIZE,
    M1_MODEL_PATH,
    M2_MODEL_PATH,
    STANDARD_FREQS,
)
from active_rl.reward import task_reward_v11


class LocalEnsembleWorldModel:
    """v9 local world model wrapper.

    It keeps the v8 EnsembleWorldModel interface while allowing the paper-facing
    M1 Chebyshev-rational MIONet to be used instead of the M2 kernel head.
    """

    def __init__(self, model_kind="m1", model_path=None, num_models=None,
                 lr=1e-5, init_noise=1e-2):
        from active_rl.networks import MIONet, MIONetKernel, PhysicsBranch

        self.model_kind = str(model_kind).lower()
        if self.model_kind not in ("m1", "m2"):
            raise ValueError(f"Unknown world model kind: {model_kind}")
        self.model_path = model_path or (M1_MODEL_PATH if self.model_kind == "m1" else M2_MODEL_PATH)
        self.num_models = int(num_models or ENSEMBLE_SIZE)
        self.PhysicsBranch = PhysicsBranch
        model_cls = MIONet if self.model_kind == "m1" else MIONetKernel

        self.models = []
        self.opts = []
        self.freq = torch.linspace(-1, 1, N_FREQ, device=DEVICE).unsqueeze(0)
        self.buffer = deque(maxlen=1000)
        state = torch.load(self.model_path, map_location=DEVICE, weights_only=True)

        for idx in range(self.num_models):
            model = model_cls(order=16, feat_dim=512, basis_type="chebyshev",
                              trunk_mode="basic", fusion_type="hadamard", dropout_p=0.0)
            model.load_state_dict(state)
            model.to(DEVICE).eval()
            if idx > 0:
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(init_noise * torch.randn_like(param) * idx)
            self.models.append(model)
            self.opts.append(torch.optim.AdamW(model.parameters(), lr=lr))

    def _prepare_inputs(self, geometries, theta=0.0, pol=0):
        if geometries.dim() == 3:
            geometries = geometries.unsqueeze(1)
        geometries = geometries.float().to(DEVICE)
        batch = geometries.shape[0]
        freq = self.freq.repeat(batch, 1) if batch > 1 else self.freq
        theta_t = torch.full((batch,), theta, device=DEVICE) if isinstance(theta, (int, float)) else theta.to(DEVICE)
        pol_t = torch.full((batch,), pol, device=DEVICE, dtype=torch.long) if isinstance(pol, int) else pol.to(DEVICE)
        enc = self.PhysicsBranch.encode_physics(theta_t, torch.zeros(batch, device=DEVICE), pol_t)
        return geometries, freq, enc

    def predict_stats(self, geometries, theta=0.0, pol=0):
        geometries, freq, enc = self._prepare_inputs(geometries, theta=theta, pol=pol)
        ctx = torch.no_grad() if hasattr(torch, "no_grad") else nullcontext()
        with ctx:
            preds_s11, preds_s21 = [], []
            for model in self.models:
                model.eval()
                s11, s21 = model(geometries, freq, enc)
                preds_s11.append(s11)
                preds_s21.append(s21)
            stack_s11 = torch.stack(preds_s11, dim=0)
            stack_s21 = torch.stack(preds_s21, dim=0)
            mean_s11 = stack_s11.mean(dim=0)
            mean_s21 = stack_s21.mean(dim=0)
            var_s11 = stack_s11.var(dim=0, unbiased=False).max().item()
            var_s21 = stack_s21.var(dim=0, unbiased=False).max().item()
            uncertainty = float(max(var_s11, var_s21))
            return mean_s11, mean_s21, uncertainty

    def predict(self, geometries, theta=0.0, pol=0):
        mean_s11, mean_s21, _ = self.predict_stats(geometries, theta=theta, pol=pol)
        return mean_s11, mean_s21

    def evaluate(self, M, spec):
        s11, s21, _ = self.predict_stats(
            M.unsqueeze(0).unsqueeze(0), spec.get("theta", 0.0), spec.get("pol", 0))
        reward, _, _ = task_reward_v11(
            s11.detach().cpu().numpy().flatten(),
            s21.detach().cpu().numpy().flatten(),
            STANDARD_FREQS,
            spec,
        )
        return reward

    def estimate_uncertainty(self, pattern_10x10, specs):
        full_t = torch.from_numpy(np.asarray(pattern_10x10, dtype=np.float32)).float().to(DEVICE)
        values = []
        for spec in specs:
            _, _, unc = self.predict_stats(
                full_t.unsqueeze(0).unsqueeze(0), spec.get("theta", 0.0), spec.get("pol", 0))
            values.append(float(unc))
        return max(values) if values else 0.0

    def add_to_buffer(self, pattern_10x10, s11_true, s21_true, theta=0.0, pol=0):
        self.buffer.append((
            pattern_10x10.copy(), s11_true.copy(), s21_true.copy(), float(theta), int(pol)
        ))

    def finetune(self, steps=10, batch_size=16):
        if len(self.buffer) < batch_size:
            return None
        loss_fn = nn.MSELoss()
        total = 0.0
        for model, opt in zip(self.models, self.opts):
            model.train()
            model_total = 0.0
            for _ in range(steps):
                batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
                imgs = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
                t11 = torch.tensor(np.stack([b[1][:N_FREQ] for b in batch]), dtype=torch.float32, device=DEVICE)
                t21 = torch.tensor(np.stack([b[2][:N_FREQ] for b in batch]), dtype=torch.float32, device=DEVICE)
                theta_t = torch.tensor([b[3] for b in batch], device=DEVICE)
                pol_t = torch.tensor([b[4] for b in batch], device=DEVICE, dtype=torch.long)
                freq = self.freq.repeat(len(batch), 1)
                enc = self.PhysicsBranch.encode_physics(theta_t, torch.zeros(len(batch), device=DEVICE), pol_t)
                p11, p21 = model(imgs, freq, enc)
                loss = loss_fn(p11, t11) + loss_fn(p21, t21)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model_total += loss.item()
            model.eval()
            total += model_total / steps
        return total / len(self.models)

    def save_checkpoint(self, path):
        payload = {
            "model_kind": self.model_kind,
            "source_model_path": self.model_path,
            "num_models": self.num_models,
            "buffer_size": len(self.buffer),
            "model_state_dicts": [model.state_dict() for model in self.models],
        }
        torch.save(payload, path)
