"""Inference-only LocalEnsembleWorldModel wrapper.

Loads a trained M1 MIONet checkpoint (3-model ensemble) and provides
predict_stats() / predict() for forward surrogate evaluation.

Does NOT train -- training is a separate module.
"""

import torch

from common.config import DEVICE, N_FREQ, M1_MODEL_PATH
from common.mionet import MIONet, PhysicsBranch


class LocalEnsembleWorldModel:
    """Inference-only ensemble of M1 MIONet models.

    Loads a single trained checkpoint, instantiates ``num_models`` copies
    (with progressively-scaled noise for diversity), and exposes the
    standard ensemble-variance uncertainty estimate via ``predict_stats``.

    Usage::

        wm = LocalEnsembleWorldModel(model_path="checkpoint.pth")
        s11, s21, unc = wm.predict_stats(geometries, theta=30.0, pol=0)
        s11, s21 = wm.predict(geometries, theta=30.0, pol=0)
    """

    def __init__(self, model_kind="m1", model_path=None, num_models=3):
        if model_kind != "m1":
            raise ValueError(
                f"LocalEnsembleWorldModel only supports model_kind='m1', got '{model_kind}'"
            )
        self.model_kind = model_kind
        self.model_path = model_path or M1_MODEL_PATH
        self.num_models = int(num_models)

        self.freq = torch.linspace(-1, 1, N_FREQ, device=DEVICE).unsqueeze(0)

        state = torch.load(self.model_path, map_location=DEVICE, weights_only=True)

        self.models = []
        for idx in range(self.num_models):
            model = MIONet(
                order=16, feat_dim=512, basis_type="chebyshev",
                trunk_mode="basic", fusion_type="hadamard", dropout_p=0.0,
            )
            model.load_state_dict(state)
            model.to(DEVICE).eval()

            # Add progressively-scaled noise for ensemble diversity
            if idx > 0:
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(1e-2 * torch.randn_like(param) * idx)

            self.models.append(model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, geometries, theta=0.0, pol=0):
        """Normalise and encode geometry / physics / frequency inputs.

        Parameters
        ----------
        geometries : torch.Tensor
            Shape ``(B, H, W)``, ``(B, 1, H, W)``, or ``(1, H, W)``.
        theta : float or torch.Tensor
            Incident angle in degrees (scalar = same for batch; tensor = per-element).
        pol : int or torch.Tensor
            Polarisation mode: 0 = TE, 1 = TM (scalar = same for batch; tensor = per-element).

        Returns
        -------
        geometries : torch.Tensor  [B, 1, H, W]
        freq : torch.Tensor        [B, N_FREQ] or [1, N_FREQ]
        enc : torch.Tensor         [B, 6] physics encoding
        """
        if geometries.dim() == 3:
            geometries = geometries.unsqueeze(1)
        geometries = geometries.float().to(DEVICE)
        batch = geometries.shape[0]

        freq = self.freq.repeat(batch, 1) if batch > 1 else self.freq

        theta_t = (
            torch.full((batch,), theta, device=DEVICE)
            if isinstance(theta, (int, float))
            else theta.to(DEVICE)
        )
        pol_t = (
            torch.full((batch,), pol, device=DEVICE, dtype=torch.long)
            if isinstance(pol, int)
            else pol.to(DEVICE)
        )

        enc = PhysicsBranch.encode_physics(
            theta_t, torch.zeros(batch, device=DEVICE), pol_t
        )
        return geometries, freq, enc

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_stats(self, geometries, theta=0.0, pol=0):
        """Ensemble mean prediction + per-pixel max-variance uncertainty.

        Returns
        -------
        mean_s11 : torch.Tensor  [B, N_FREQ]
        mean_s21 : torch.Tensor  [B, N_FREQ]
        uncertainty : float
            ``max(var_s11.max(), var_s21.max())`` across the ensemble.
        """
        geometries, freq, enc = self._prepare_inputs(
            geometries, theta=theta, pol=pol
        )

        preds_s11, preds_s21 = [], []
        for model in self.models:
            model.eval()
            s11, s21 = model(geometries, freq, enc)
            preds_s11.append(s11)
            preds_s21.append(s21)

        stack_s11 = torch.stack(preds_s11, dim=0)  # [num_models, B, N_FREQ]
        stack_s21 = torch.stack(preds_s21, dim=0)

        mean_s11 = stack_s11.mean(dim=0)
        mean_s21 = stack_s21.mean(dim=0)

        var_s11 = stack_s11.var(dim=0, unbiased=False).max().item()
        var_s21 = stack_s21.var(dim=0, unbiased=False).max().item()
        uncertainty = float(max(var_s11, var_s21))

        return mean_s11, mean_s21, uncertainty

    @torch.no_grad()
    def predict(self, geometries, theta=0.0, pol=0):
        """Ensemble mean prediction (no uncertainty).

        Returns
        -------
        s11 : torch.Tensor  [B, N_FREQ]
        s21 : torch.Tensor  [B, N_FREQ]
        """
        mean_s11, mean_s21, _ = self.predict_stats(
            geometries, theta=theta, pol=pol
        )
        return mean_s11, mean_s21
