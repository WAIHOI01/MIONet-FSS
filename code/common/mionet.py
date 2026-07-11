"""MIONet (Multiple-Input Operator Network) architecture.

Three-branch architecture:
  Branch 1 (Spatial):   CNN encoder for 10x10 geometry image -> spatial features
  Branch 2 (Physics):   MLP encoder for [theta, phi, pol] -> physics features
  Trunk (Frequency):    Chebyshev basis + MLP -> frequency basis functions

Fusion: Hadamard product of branch features -> rational output -> S11, S21

Polarisation handling: driven entirely through the Physics Branch
[pol_TE, pol_TM] one-hot input, replacing hard-coded rot90.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import build_freq_basis


# ==========================================
# Helper
# ==========================================

class ResBlock(nn.Module):
    """Residual block for the multi-scale spatial backbone."""
    def __init__(self, channels: int, padding_mode: str = "circular"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# ==========================================
# Branch 1: Spatial topology (CNN multi-scale backbone)
# ==========================================

class SpatialBranch(nn.Module):
    """CNN encoder for 10x10 FSS geometry images.

    Three-stage multi-scale design:
      Stage1: keeps 10x10 resolution -> 64 channels
      Stage2: stride=2 downsample  -> 128 channels
      Stage3: stride=2 downsample  -> 256 channels
      Fusion:  three-stage GAP -> concat -> Linear -> feat_dim output
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 512,
                 dropout_p: float = 0.0, padding_mode: str = "circular"):
        super().__init__()
        self.feat_dim = feat_dim

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResBlock(64, padding_mode),
            ResBlock(64, padding_mode),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResBlock(128, padding_mode),
            ResBlock(128, padding_mode),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResBlock(256, padding_mode),
            ResBlock(256, padding_mode),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fuse = nn.Sequential(
            nn.Linear(64 + 128 + 256, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, 10, 10] geometry image
        Returns:
            [B, feat_dim] spatial features
        """
        f1 = self.stage1(x)       # [B, 64, 10, 10]
        f2 = self.stage2(f1)      # [B, 128, 5, 5]
        f3 = self.stage3(f2)      # [B, 256, 3, 3]

        g1 = self.pool(f1).flatten(1)  # [B, 64]
        g2 = self.pool(f2).flatten(1)  # [B, 128]
        g3 = self.pool(f3).flatten(1)  # [B, 256]

        return self.fuse(torch.cat([g1, g2, g3], dim=1))


# ==========================================
# Branch 2: Physics state (MLP)
# ==========================================

class PhysicsBranch(nn.Module):
    """MLP encoder for incident physics state [theta, phi, polarisation].

    Input encoding:
      - sin(theta), cos(theta)  -- angular periodic encoding
      - sin(phi),   cos(phi)    -- azimuth periodic encoding (default 0)
      - pol_TE,     pol_TM      -- polarisation one-hot: [1,0]=TE, [0,1]=TM

    This replaces hard-coded rot90: the same network outputs the
    appropriate S-parameter prediction based on the pol input.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 256,
                 feat_dim: int = 512, dropout_p: float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, feat_dim),
        )

    @staticmethod
    def encode_physics(theta_deg: torch.Tensor,
                       phi_deg: torch.Tensor = None,
                       pol_mode: torch.Tensor = None) -> torch.Tensor:
        """Build standard physics encoding for the Physics Branch.

        Args:
            theta_deg: [B] or scalar, incident angle (degrees)
            phi_deg:   [B] or scalar or None, azimuth angle (degrees), default 0
            pol_mode:  [B] or scalar or None, polarisation (0=TE, 1=TM), default TE

        Returns:
            [B, 6] physics state encoding
        """
        if theta_deg.dim() == 0:
            theta_deg = theta_deg.unsqueeze(0)
        batch = theta_deg.shape[0]

        theta_rad = torch.deg2rad(theta_deg)
        sin_theta = torch.sin(theta_rad).unsqueeze(-1)
        cos_theta = torch.cos(theta_rad).unsqueeze(-1)

        if phi_deg is None:
            phi_deg = torch.zeros_like(theta_deg)
        elif phi_deg.dim() == 0:
            phi_deg = phi_deg.unsqueeze(0)
        phi_rad = torch.deg2rad(phi_deg)
        sin_phi = torch.sin(phi_rad).unsqueeze(-1)
        cos_phi = torch.cos(phi_rad).unsqueeze(-1)

        if pol_mode is None:
            pol_mode = torch.zeros_like(theta_deg)  # default TE
        elif pol_mode.dim() == 0:
            pol_mode = pol_mode.unsqueeze(0)
        pol_te = (pol_mode == 0).float().unsqueeze(-1)
        pol_tm = (pol_mode == 1).float().unsqueeze(-1)

        return torch.cat([sin_theta, cos_theta, sin_phi, cos_phi, pol_te, pol_tm], dim=-1)

    def forward(self, physics_encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_encoding: [B, input_dim] physics encoding (from encode_physics)
        Returns:
            [B, feat_dim] physics condition features
        """
        return self.net(physics_encoding)


# ==========================================
# Trunk: frequency basis + MLP
# ==========================================

class ChebyshevTrunk(nn.Module):
    """Frequency trunk network: Chebyshev orthogonal basis -> MLP -> frequency
    basis functions.

    Supports two modes:
      - "basic":    frequency f only
      - "floquet":  frequency f + Floquet wavenumbers kx, ky (for oblique incidence)
    """

    def __init__(self, order: int = 16, feat_dim: int = 512,
                 hidden_dim: int = 256, basis_type: str = "chebyshev",
                 trunk_mode: str = "basic", dropout_p: float = 0.0):
        super().__init__()
        self.order = order
        self.feat_dim = feat_dim
        self.basis_type = basis_type
        self.trunk_mode = trunk_mode

        # input dimension: order+1 (basic) or order+1+2 (floquet: +kx, ky)
        input_dim = order + 1
        if trunk_mode == "floquet":
            input_dim += 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, feat_dim),
        )

    def build_basis(self, x_freq: torch.Tensor,
                    theta_deg: torch.Tensor = None) -> torch.Tensor:
        """Build frequency basis (with optional Floquet wavenumber augmentation).

        Args:
            x_freq:    [B, F] normalised frequency
            theta_deg: [B] incident angle (required for floquet mode)

        Returns:
            [B, F, input_dim] basis tensor
        """
        if x_freq.dim() == 1:
            x_freq = x_freq.unsqueeze(0)

        # build_freq_basis returns (..., K+1, F); transpose to (..., F, K+1)
        freq_basis = build_freq_basis(x_freq, self.order, self.basis_type)
        freq_basis = freq_basis.transpose(-1, -2)  # [B, F, order+1]

        if self.trunk_mode == "floquet" and theta_deg is not None:
            # Floquet wavenumbers: kx = k0 * sin(theta) * cos(phi),
            #                      ky = k0 * sin(theta) * sin(phi)
            # Under normalised frequency: kx_norm = sin(theta), ky_norm = 0
            if theta_deg.dim() == 0:
                theta_deg = theta_deg.unsqueeze(0)
            theta_rad = torch.deg2rad(theta_deg)
            kx = (torch.sin(theta_rad) * torch.cos(torch.zeros_like(theta_rad))
                  ).unsqueeze(-1).unsqueeze(-1)
            ky = (torch.sin(theta_rad) * torch.sin(torch.zeros_like(theta_rad))
                  ).unsqueeze(-1).unsqueeze(-1)
            kx = kx.expand(-1, freq_basis.shape[1], 1)
            ky = ky.expand(-1, freq_basis.shape[1], 1)
            freq_basis = torch.cat([freq_basis, kx, ky], dim=-1)

        return freq_basis

    def forward(self, freq_basis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq_basis: [B, F, input_dim]
        Returns:
            [B, F, feat_dim] frequency basis functions
        """
        return self.net(freq_basis)


# ==========================================
# Fusion and output
# ==========================================

class HadamardFusion(nn.Module):
    """Hadamard-product fusion with rational output head.

    Fusion strategies:
      - "hadamard":  v_fused = v1 * v2          (multiplicative, default)
      - "additive":  v_fused = v1 + v2           (additive)
      - "bilinear":  v_fused = v1 @ W @ v2.T     (bilinear)

    Output: rational function S(f) = sigmoid(num / (1 + |den|))
    """

    def __init__(self, feat_dim: int = 512, fusion_type: str = "hadamard",
                 num_s_params: int = 2, order: int = 16):
        super().__init__()
        self.feat_dim = feat_dim
        self.fusion_type = fusion_type
        self.order = order

        if fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(feat_dim, feat_dim, feat_dim)

        # Each S-parameter needs 2*order+1 coefficients (numerator: order+1, denominator: order)
        coeffs_per_s = 2 * order + 1
        total_coeffs = num_s_params * coeffs_per_s
        self.to_coeffs = nn.Linear(feat_dim, total_coeffs)

    def forward(self, v_spatial: torch.Tensor, v_physics: torch.Tensor,
                v_trunk: torch.Tensor) -> tuple:
        """
        Args:
            v_spatial: [B, feat_dim]    spatial features
            v_physics: [B, feat_dim]    physics condition features
            v_trunk:   [B, F, feat_dim] frequency basis functions

        Returns:
            (s11_pred, s21_pred): each [B, F]
        """
        # 1. Spatial-physics fusion
        if self.fusion_type == "hadamard":
            v_fused = v_spatial * v_physics  # [B, feat_dim]
        elif self.fusion_type == "additive":
            v_fused = v_spatial + v_physics
        elif self.fusion_type == "bilinear":
            v_fused = self.bilinear(v_spatial, v_physics)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # 2. Generate rational coefficients
        coeffs = self.to_coeffs(v_fused)  # [B, total_coeffs]

        # 3. Evaluate rational function
        return self._eval_rational(coeffs, v_trunk)

    def _eval_rational(self, coeffs: torch.Tensor,
                       v_trunk: torch.Tensor) -> tuple:
        """Evaluate rational function: S(f) = sigmoid(num / (1 + |den|)).

        Args:
            coeffs:   [B, total_coeffs]
            v_trunk:  [B, F, feat_dim]

        Returns:
            (s11, s21): each [B, F]
        """
        B, F, D = v_trunk.shape
        coeffs_per_s = 2 * self.order + 1

        c_s11 = coeffs[:, :coeffs_per_s]       # [B, coeffs_per_s]
        c_s21 = coeffs[:, coeffs_per_s:]       # [B, coeffs_per_s]

        # numerator: first order+1 trunk features
        # denominator: next order trunk features (skip constant term)
        num_basis = v_trunk[:, :, :self.order + 1]      # [B, F, order+1]
        den_basis = v_trunk[:, :, 1:self.order + 1]     # [B, F, order]

        num_s11 = torch.bmm(num_basis, c_s11[:, :self.order + 1].unsqueeze(-1)).squeeze(-1)
        den_s11 = torch.bmm(den_basis, c_s11[:, self.order + 1:].unsqueeze(-1)).squeeze(-1)

        num_s21 = torch.bmm(num_basis, c_s21[:, :self.order + 1].unsqueeze(-1)).squeeze(-1)
        den_s21 = torch.bmm(den_basis, c_s21[:, self.order + 1:].unsqueeze(-1)).squeeze(-1)

        out_s11 = num_s11 / (1.0 + torch.abs(den_s11) + 1e-6)
        out_s21 = num_s21 / (1.0 + torch.abs(den_s21) + 1e-6)

        return torch.sigmoid(out_s11), torch.sigmoid(out_s21)


# ==========================================
# Full MIONet model
# ==========================================

class MIONet(nn.Module):
    """Complete MIONet: 3-branch + Hadamard fusion + rational output.

    Polarisation selection:
      Use PhysicsBranch.encode_physics() to build an encoding that includes
      the pol input.  Call the same model twice (pol_TE / pol_TM) to obtain
      dual-polarisation predictions -- no hard-coded rot90 needed.

    Usage:
        model = MIONet(order=16, basis_type="chebyshev")
        phys_enc = PhysicsBranch.encode_physics(theta, phi, pol_mode)
        s11, s21 = model(x_img, x_freq, phys_enc)
    """

    def __init__(self, order: int = 16, feat_dim: int = 512,
                 basis_type: str = "chebyshev", trunk_mode: str = "basic",
                 fusion_type: str = "hadamard", dropout_p: float = 0.0):
        super().__init__()
        self.order = order
        self.feat_dim = feat_dim

        self.spatial_branch = SpatialBranch(
            in_channels=1, feat_dim=feat_dim, dropout_p=dropout_p)
        self.physics_branch = PhysicsBranch(
            input_dim=6, feat_dim=feat_dim, dropout_p=dropout_p)
        self.trunk = ChebyshevTrunk(
            order=order, feat_dim=feat_dim, basis_type=basis_type,
            trunk_mode=trunk_mode, dropout_p=dropout_p)
        self.fusion = HadamardFusion(
            feat_dim=feat_dim, fusion_type=fusion_type,
            num_s_params=2, order=order)

    def build_physics_encoding(self, theta_deg: torch.Tensor,
                               phi_deg: torch.Tensor = None,
                               pol_mode: torch.Tensor = None) -> torch.Tensor:
        """Convenience: build physics state encoding."""
        return PhysicsBranch.encode_physics(theta_deg, phi_deg, pol_mode)

    def forward(self, x_img: torch.Tensor, x_freq: torch.Tensor,
                physics_enc: torch.Tensor) -> tuple:
        """
        Args:
            x_img:       [B, 1, 10, 10] geometry image
            x_freq:      [B, F] normalised frequency
            physics_enc: [B, 6] physics state encoding

        Returns:
            (s11_pred, s21_pred): each [B, F]
        """
        if x_freq.dim() == 1:
            x_freq = x_freq.unsqueeze(0)
        batch = x_img.shape[0]
        if x_freq.shape[0] == 1 and batch > 1:
            x_freq = x_freq.repeat(batch, 1)

        v1 = self.spatial_branch(x_img)
        v2 = self.physics_branch(physics_enc)
        freq_basis = self.trunk.build_basis(x_freq)
        v3 = self.trunk(freq_basis)

        return self.fusion(v1, v2, v3)

    def forward_dual_pol(self, x_img: torch.Tensor, x_freq: torch.Tensor,
                         theta_deg: torch.Tensor,
                         phi_deg: torch.Tensor = None) -> tuple:
        """Convenience: single forward pass returning TE and TM predictions.

        Args:
            x_img:      [B, 1, 10, 10]
            x_freq:     [B, F]
            theta_deg:  [B]
            phi_deg:    [B] or None

        Returns:
            (te_s11, te_s21, tm_s11, tm_s21): each [B, F]
        """
        if phi_deg is None:
            phi_deg = torch.zeros_like(theta_deg)

        batch = theta_deg.shape[0]
        pol_te = torch.zeros(batch, device=theta_deg.device)  # 0 = TE
        pol_tm = torch.ones(batch, device=theta_deg.device)   # 1 = TM

        enc_te = PhysicsBranch.encode_physics(theta_deg, phi_deg, pol_te)
        enc_tm = PhysicsBranch.encode_physics(theta_deg, phi_deg, pol_tm)

        te_s11, te_s21 = self.forward(x_img, x_freq, enc_te)
        tm_s11, tm_s21 = self.forward(x_img, x_freq, enc_tm)

        return te_s11, te_s21, tm_s11, tm_s21
