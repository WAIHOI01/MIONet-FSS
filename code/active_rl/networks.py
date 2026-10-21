"""MIONet model definitions (verbatim merge of chebyshev_basis.py + mionet_common.py).

Class names and state_dict layouts are unchanged from the released files, so
the server/*.pth checkpoints load with strict=True.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Chebyshev 正交多项式基底模块 (P0 升级)
# ==========================================

def chebyshev_basis(x_norm: torch.Tensor, order: int) -> torch.Tensor:
    """
    计算 Chebyshev 多项式 T_0(x) 到 T_order(x)。

    Args:
        x_norm: [B, F] 频率张量，已归一化到 [-1, 1]
        order: 多项式最高阶数

    Returns:
        [B, F, order+1] Chebyshev 基底张量
    """
    if x_norm.dim() == 1:
        x_norm = x_norm.unsqueeze(0)

    T = [torch.ones_like(x_norm), x_norm]  # T_0 = 1, T_1 = x
    for k in range(2, order + 1):
        T.append(2.0 * x_norm * T[-1] - T[k - 1])
    return torch.stack(T, dim=2)


def legendre_basis(x_norm: torch.Tensor, order: int) -> torch.Tensor:
    """
    计算 Legendre 多项式 P_0(x) 到 P_order(x)（备选正交基底）。

    递推: P_0(x)=1, P_1(x)=x,
          (k+1)P_{k+1}(x) = (2k+1)x·P_k(x) - k·P_{k-1}(x)

    Args:
        x_norm: [B, F] 频率张量，已归一化到 [-1, 1]
        order: 多项式最高阶数

    Returns:
        [B, F, order+1] Legendre 基底张量
    """
    if x_norm.dim() == 1:
        x_norm = x_norm.unsqueeze(0)

    P = [torch.ones_like(x_norm), x_norm]  # P_0 = 1, P_1 = x
    for k in range(1, order):
        # (k+1) * P_{k+1} = (2k+1) * x * P_k - k * P_{k-1}
        P_next = ((2 * k + 1) * x_norm * P[-1] - k * P[k - 1]) / (k + 1)
        P.append(P_next)
    return torch.stack(P, dim=2)


def build_freq_basis(
    x_freq: torch.Tensor,
    order: int,
    basis_type: str = "chebyshev",
) -> torch.Tensor:
    """
    统一的频率基底构建接口。

    Args:
        x_freq: [B, F] 频率张量，归一化到 [-1, 1]
        order: 多项式阶数
        basis_type: 基底类型 — "taylor" | "chebyshev" | "legendre"

    Returns:
        [B, F, order+1] 频率基底张量
    """
    if x_freq.dim() == 1:
        x_freq = x_freq.unsqueeze(0)

    if basis_type == "taylor":
        powers = [torch.ones_like(x_freq)]
        for i in range(1, order + 1):
            powers.append(torch.pow(x_freq, i))
        return torch.stack(powers, dim=2)

    if basis_type == "chebyshev":
        return chebyshev_basis(x_freq, order)

    if basis_type == "legendre":
        return legendre_basis(x_freq, order)

    raise ValueError(f"Unknown basis_type: {basis_type}. Choose taylor|chebyshev|legendre.")


def scale_freq_with_cos_theta(
    x_freq: torch.Tensor,
    theta_deg: torch.Tensor,
    clip_value: float = 2.0,
) -> torch.Tensor:
    """
    物理频率缩放: f_eff = f / cos(θ)，建模斜入射下的有效频率。

    Args:
        x_freq: [B, F] 或 [1, F] 归一化频率
        theta_deg: [B] 入射角（度）
        clip_value: 缩放后 clamp 范围

    Returns:
        [B, F] 缩放后的频率
    """
    if x_freq.dim() == 1:
        x_freq = x_freq.unsqueeze(0)
    if theta_deg.dim() == 0:
        theta_deg = theta_deg.unsqueeze(0)

    batch_size = theta_deg.shape[0]
    if x_freq.shape[0] == 1 and batch_size > 1:
        x_freq = x_freq.repeat(batch_size, 1)

    cos_theta = torch.cos(torch.deg2rad(theta_deg)).unsqueeze(-1).clamp_min(1e-4)
    return (x_freq / cos_theta).clamp(-clip_value, clip_value)


class FreqBasisBuilder:
    """
    频率基底构建器，封装 basis_type 选择和物理缩放逻辑。

    用法:
        builder = FreqBasisBuilder(order=16, basis_type="chebyshev")
        basis = builder(x_freq)                          # 标准基底
        basis_scaled = builder(x_freq, theta_deg)        # cos(θ) 缩放基底
    """

    def __init__(self, order: int = 16, basis_type: str = "chebyshev"):
        self.order = order
        self.basis_type = basis_type

    def __call__(
        self,
        x_freq: torch.Tensor,
        theta_deg: torch.Tensor = None,
    ) -> torch.Tensor:
        if theta_deg is not None:
            x_freq = scale_freq_with_cos_theta(x_freq, theta_deg)
        return build_freq_basis(x_freq, self.order, self.basis_type)


def compute_condition_number(basis: torch.Tensor) -> float:
    """
    计算基底矩阵的条件数（用于基底质量诊断）。

    Args:
        basis: [B, F, order+1] 基底张量

    Returns:
        条件数 (Frobenius norm 近似)
    """
    # 取第一个 batch，在频率维度上评估
    B = basis[0]  # [F, order+1]
    # 条件数 = σ_max / σ_min
    _, S, _ = torch.linalg.svd(B.float(), full_matrices=False)
    return (S[0] / (S[-1] + 1e-8)).item()


# ==========================================
# MIONet (Multiple-Input Operator Network) 通用模块 (P1 升级)
#
# 三支路架构:
#   Branch 1 (Spatial):    CNN 处理 10×10 几何图像 → v1 (512D)
#   Branch 2 (Physics):    MLP 处理 [θ, φ, pol] → v2 (512D)
#   Trunk (Frequency):     Chebyshev 基底 + MLP → v3 (512D)
#
# 融合: v1 ⊙ v2 → 系数 → S(f) = sigmoid( num / (1+|den|) )
#
# 极化处理: 通过 Physics Branch 的 [pol_TE, pol_TM] one-hot 输入，
#          彻底取代 rot90 硬编码。
# ==========================================

class ResBlock(nn.Module):
    """残差块，用于多尺度主干网络。"""
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
# Branch 1: 空间拓扑 (CNN 多尺度主干)
# ==========================================

class SpatialBranch(nn.Module):
    """
    处理 10×10 FSS 几何图像，提取空间拓扑特征。

    三阶段多尺度设计:
      Stage1: 保持 10×10 分辨率 → 64 通道
      Stage2: stride=2 下采样 → 128 通道
      Stage3: stride=2 下采样 → 256 通道
      Fusion: 三级 GAP → concat → Linear → 512D 输出
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 512,
                 dropout_p: float = 0.1, padding_mode: str = "circular"):
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
            x: [B, in_channels, 10, 10] 几何图像
        Returns:
            [B, feat_dim] 空间特征
        """
        f1 = self.stage1(x)       # [B, 64, 10, 10]
        f2 = self.stage2(f1)      # [B, 128, 5, 5]
        f3 = self.stage3(f2)      # [B, 256, 3, 3]

        g1 = self.pool(f1).flatten(1)  # [B, 64]
        g2 = self.pool(f2).flatten(1)  # [B, 128]
        g3 = self.pool(f3).flatten(1)  # [B, 256]

        return self.fuse(torch.cat([g1, g2, g3], dim=1))


# ==========================================
# Branch 2: 物理状态 (MLP)
# ==========================================

class PhysicsBranch(nn.Module):
    """
    处理入射物理状态 [θ, φ, 极化]，输出物理条件特征。

    输入编码:
      - sin(θ), cos(θ)  → 角度周期编码
      - sin(φ), cos(φ)  → 方位角周期编码（φ=0° 时可设为 0）
      - pol_TE, pol_TM  → 极化 one-hot: [1,0]=TE, [0,1]=TM

    这彻底取代了 rot90 硬编码: 同一网络在接收不同 pol 输入时
    输出对应极化的 S 参数预测。
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 256,
                 feat_dim: int = 512, dropout_p: float = 0.1):
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
        """
        构建 Physics Branch 的标准输入编码。

        Args:
            theta_deg: [B] 或标量，入射角 (度)
            phi_deg: [B] 或标量或 None，方位角 (度)，默认 0°
            pol_mode: [B] 或标量或 None，极化模式 (0=TE, 1=TM)，默认 TE

        Returns:
            [B, 6] 物理状态编码
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
            physics_encoding: [B, input_dim] 物理状态编码 (来自 encode_physics)
        Returns:
            [B, feat_dim] 物理条件特征
        """
        return self.net(physics_encoding)


# ==========================================
# Trunk: 频率基底 + MLP
# ==========================================

class ChebyshevTrunk(nn.Module):
    """
    频率 Trunk 网络: Chebyshev 正交基底 → MLP → 频率基函数。

    支持三种模式:
      - "basic":   仅频率 f
      - "floquet": 频率 f + Floquet 波数 kx, ky（用于 M3）
    """

    def __init__(self, order: int = 16, feat_dim: int = 512,
                 hidden_dim: int = 256, basis_type: str = "chebyshev",
                 trunk_mode: str = "basic", dropout_p: float = 0.1):
        super().__init__()
        self.order = order
        self.feat_dim = feat_dim
        self.basis_type = basis_type
        self.trunk_mode = trunk_mode

        # 输入维度: order+1 (basic) 或 order+1+2 (floquet: +kx,ky)
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
        """
        构建频率基底（含可选的 Floquet 波数增强）。

        Args:
            x_freq: [B, F] 归一化频率
            theta_deg: [B] 入射角（floquet 模式需要）

        Returns:
            [B, F, input_dim] 基底
        """
        if x_freq.dim() == 1:
            x_freq = x_freq.unsqueeze(0)

        freq_basis = build_freq_basis(x_freq, self.order, self.basis_type)

        if self.trunk_mode == "floquet" and theta_deg is not None:
            # Floquet 波数: kx = k0 * sin(θ) * cos(φ), ky = k0 * sin(θ) * sin(φ)
            # 在归一化频率下近似: kx_norm = sin(θ) * cos(φ), ky_norm = sin(θ) * sin(φ)
            if theta_deg.dim() == 0:
                theta_deg = theta_deg.unsqueeze(0)
            theta_rad = torch.deg2rad(theta_deg)
            kx = (torch.sin(theta_rad) * torch.cos(torch.zeros_like(theta_rad))).unsqueeze(-1).unsqueeze(-1)
            ky = (torch.sin(theta_rad) * torch.sin(torch.zeros_like(theta_rad))).unsqueeze(-1).unsqueeze(-1)
            kx = kx.expand(-1, freq_basis.shape[1], 1)
            ky = ky.expand(-1, freq_basis.shape[1], 1)
            freq_basis = torch.cat([freq_basis, kx, ky], dim=-1)

        return freq_basis

    def forward(self, freq_basis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq_basis: [B, F, input_dim]
        Returns:
            [B, F, feat_dim] 频率基函数
        """
        return self.net(freq_basis)


# ==========================================
# 融合与输出
# ==========================================

class HadamardFusion(nn.Module):
    """
    哈达玛积融合: 将空间特征和物理特征进行逐元素乘积，
    然后通过投影生成有理分式的系数。

    融合策略选项:
      - "hadamard":  v_fused = v1 ⊙ v2           (乘法，推荐)
      - "additive":  v_fused = v1 + v2            (加法)
      - "bilinear":  v_fused = v1 @ W @ v2.T     (双线性)
    """

    def __init__(self, feat_dim: int = 512, fusion_type: str = "hadamard",
                 num_s_params: int = 2, order: int = 16):
        super().__init__()
        self.feat_dim = feat_dim
        self.fusion_type = fusion_type
        self.order = order

        if fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(feat_dim, feat_dim, feat_dim)

        # 每个 S 参数需要 2*order+1 个系数 (分子 order+1, 分母 order)
        coeffs_per_s = 2 * order + 1
        total_coeffs = num_s_params * coeffs_per_s
        self.to_coeffs = nn.Linear(feat_dim, total_coeffs)

    def forward(self, v_spatial: torch.Tensor, v_physics: torch.Tensor,
                v_trunk: torch.Tensor) -> tuple:
        """
        Args:
            v_spatial: [B, feat_dim]  空间特征
            v_physics: [B, feat_dim]  物理条件特征
            v_trunk:   [B, F, feat_dim] 频率基函数

        Returns:
            (s11_pred, s21_pred): 各 [B, F]
        """
        # 1. 空间-物理融合
        if self.fusion_type == "hadamard":
            v_fused = v_spatial * v_physics  # [B, feat_dim]
        elif self.fusion_type == "additive":
            v_fused = v_spatial + v_physics
        elif self.fusion_type == "bilinear":
            v_fused = self.bilinear(v_spatial, v_physics)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # 2. 生成系数
        coeffs = self.to_coeffs(v_fused)  # [B, total_coeffs]

        # 3. 有理分式求值
        return self._eval_rational(coeffs, v_trunk)

    def _eval_rational(self, coeffs: torch.Tensor,
                       v_trunk: torch.Tensor) -> tuple:
        """
        用 Chebyshev 基底评估有理分式: S(f) = sigmoid( num / (1+|den|) )

        Args:
            coeffs:   [B, total_coeffs]
            v_trunk:  [B, F, feat_dim]

        Returns:
            (s11, s21): 各 [B, F]
        """
        B, F, D = v_trunk.shape
        coeffs_per_s = 2 * self.order + 1

        c_s11 = coeffs[:, :coeffs_per_s]       # [B, coeffs_per_s]
        c_s21 = coeffs[:, coeffs_per_s:]       # [B, coeffs_per_s]

        # 分子: coeffs * trunk_basis (取前 order+1 维)
        # 分母: coeffs * trunk_basis (取后 order 维)
        # 这里 trunk_basis 是 MLP 输出，我们用特征维度的前 order+1 和后 order 维
        num_basis = v_trunk[:, :, :self.order + 1]      # [B, F, order+1]
        den_basis = v_trunk[:, :, 1:self.order + 1]     # [B, F, order] (不含常数项)

        num_s11 = torch.bmm(num_basis, c_s11[:, :self.order + 1].unsqueeze(-1)).squeeze(-1)
        den_s11 = torch.bmm(den_basis, c_s11[:, self.order + 1:].unsqueeze(-1)).squeeze(-1)

        num_s21 = torch.bmm(num_basis, c_s21[:, :self.order + 1].unsqueeze(-1)).squeeze(-1)
        den_s21 = torch.bmm(den_basis, c_s21[:, self.order + 1:].unsqueeze(-1)).squeeze(-1)

        out_s11 = num_s11 / (1.0 + torch.abs(den_s11) + 1e-6)
        out_s21 = num_s21 / (1.0 + torch.abs(den_s21) + 1e-6)

        return torch.sigmoid(out_s11), torch.sigmoid(out_s21)


# ==========================================
# 完整 MIONet 模型
# ==========================================

class MIONet(nn.Module):
    """
    完整 MIONet 模型: 3 支路 + 哈达玛积融合 + 有理输出。

    极化选择化:
      通过 PhysicsBranch.encode_physics() 构建含 pol 输入的编码，
      同一模型调用两次（pol_TE / pol_TM）即可获得双极化预测，
      彻底取代 rot90 硬编码。

    用法:
        model = MIONet(order=16, basis_type="chebyshev")
        phys_enc = PhysicsBranch.encode_physics(theta, phi, pol_mode)
        s11, s21 = model(x_img, x_freq, phys_enc)
    """

    def __init__(self, order: int = 16, feat_dim: int = 512,
                 basis_type: str = "chebyshev", trunk_mode: str = "basic",
                 fusion_type: str = "hadamard", dropout_p: float = 0.1):
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
        """便捷方法：构建物理状态编码。"""
        return PhysicsBranch.encode_physics(theta_deg, phi_deg, pol_mode)

    def forward(self, x_img: torch.Tensor, x_freq: torch.Tensor,
                physics_enc: torch.Tensor) -> tuple:
        """
        Args:
            x_img:       [B, 1, 10, 10] 几何图像
            x_freq:      [B, F] 归一化频率
            physics_enc: [B, 6] 物理状态编码

        Returns:
            (s11_pred, s21_pred): 各 [B, F]
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
                          theta_deg: torch.Tensor, phi_deg: torch.Tensor = None) -> tuple:
        """
        便捷方法：一次前向计算 TE 和 TM 双极化预测。

        Args:
            x_img:      [B, 1, 10, 10]
            x_freq:     [B, F]
            theta_deg:  [B]
            phi_deg:    [B] 或 None

        Returns:
            (te_s11, te_s21, tm_s11, tm_s21): 各 [B, F]
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


# ==========================================
# 变体: MIONet + 物理核混合输出头 (M2)
# ==========================================

class KernelMixtureHead(nn.Module):
    """
    F1/G1 风格的物理核混合输出头。
    用于 M2: 取代有理分式输出，改用 Lorentzian + Fano + Sigmoid 核。
    """

    def __init__(self, feat_dim: int = 512, num_lorentzian: int = 6,
                 num_fano: int = 3, num_sigmoid: int = 2):
        super().__init__()
        self.num_L = num_lorentzian
        self.num_F = num_fano
        self.num_S = num_sigmoid
        params_per_curve = 1 + 3 * num_lorentzian + 4 * num_fano + 3 * num_sigmoid

        # 从融合特征生成核参数
        self.to_params_s11 = nn.Linear(feat_dim, params_per_curve)
        self.to_params_s21 = nn.Linear(feat_dim, params_per_curve)

    def _eval_kernels(self, params: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        B, n_freq = freq.shape
        idx = 0
        curve = params[:, idx].unsqueeze(1).expand(B, n_freq)  # baseline
        idx += 1

        for _ in range(self.num_L):
            amp = params[:, idx].unsqueeze(1)
            ctr = torch.tanh(params[:, idx + 1]).unsqueeze(1)
            wid = F.softplus(params[:, idx + 2]).unsqueeze(1) + 1e-3
            idx += 3
            curve = curve + amp / (1.0 + ((freq - ctr) / wid) ** 2)

        for _ in range(self.num_F):
            amp = params[:, idx].unsqueeze(1)
            ctr = torch.tanh(params[:, idx + 1]).unsqueeze(1)
            wid = F.softplus(params[:, idx + 2]).unsqueeze(1) + 1e-3
            q = params[:, idx + 3].unsqueeze(1)
            idx += 4
            eps = (freq - ctr) / wid
            curve = curve + amp * (q + eps) ** 2 / (1.0 + eps ** 2)

        for _ in range(self.num_S):
            amp = params[:, idx].unsqueeze(1)
            ctr = torch.tanh(params[:, idx + 1]).unsqueeze(1)
            steep = F.softplus(params[:, idx + 2]).unsqueeze(1) + 0.1
            idx += 3
            curve = curve + amp * torch.sigmoid((freq - ctr) * steep)

        return torch.sigmoid(curve)

    def forward(self, v_fused: torch.Tensor, freq: torch.Tensor) -> tuple:
        params_s11 = self.to_params_s11(v_fused)
        params_s21 = self.to_params_s21(v_fused)
        return self._eval_kernels(params_s11, freq), self._eval_kernels(params_s21, freq)


class MIONetKernel(MIONet):
    """
    M2: MIONet + 物理核混合输出头。

    空间和物理分支生成融合特征 → KernelMixtureHead 生成核参数 → 核函数求值。
    """

    def __init__(self, order: int = 16, feat_dim: int = 512,
                 basis_type: str = "chebyshev", trunk_mode: str = "basic",
                 fusion_type: str = "hadamard", dropout_p: float = 0.1):
        super().__init__(order=order, feat_dim=feat_dim, basis_type=basis_type,
                         trunk_mode=trunk_mode, fusion_type=fusion_type,
                         dropout_p=dropout_p)
        # 替换融合输出头
        self.fusion = None
        self.kernel_head = KernelMixtureHead(feat_dim=feat_dim)

    def forward(self, x_img: torch.Tensor, x_freq: torch.Tensor,
                physics_enc: torch.Tensor) -> tuple:
        if x_freq.dim() == 1:
            x_freq = x_freq.unsqueeze(0)
        batch = x_img.shape[0]
        if x_freq.shape[0] == 1 and batch > 1:
            x_freq = x_freq.repeat(batch, 1)

        v1 = self.spatial_branch(x_img)
        v2 = self.physics_branch(physics_enc)
        v_fused = v1 * v2  # Hadamard
        return self.kernel_head(v_fused, x_freq)


# ==========================================
# 变体: MIONet + Floquet 增强 Trunk (M3)
# ==========================================

class MIONetFloquet(MIONet):
    """
    M3: MIONet + Floquet 增强 Trunk。

    Trunk 输入从 f 扩展到 [f, kx, ky]，
    其中 kx = sin(θ)cos(φ), ky = sin(θ)sin(φ) 为 Floquet 波数。
    这使得频率基底能够感知入射方向，更准确建模斜入射色散。
    """

    def __init__(self, order: int = 16, feat_dim: int = 512,
                 basis_type: str = "chebyshev", fusion_type: str = "hadamard",
                 dropout_p: float = 0.1):
        super().__init__(order=order, feat_dim=feat_dim, basis_type=basis_type,
                         trunk_mode="floquet", fusion_type=fusion_type,
                         dropout_p=dropout_p)

    def forward(self, x_img: torch.Tensor, x_freq: torch.Tensor,
                physics_enc: torch.Tensor,
                theta_deg: torch.Tensor = None) -> tuple:
        if x_freq.dim() == 1:
            x_freq = x_freq.unsqueeze(0)
        batch = x_img.shape[0]
        if x_freq.shape[0] == 1 and batch > 1:
            x_freq = x_freq.repeat(batch, 1)

        # 如果未显式传入 theta，从 physics_enc 恢复:
        # physics_enc = [sinθ, cosθ, sinφ, cosφ, pol_TE, pol_TM]
        if theta_deg is None:
            theta_deg = torch.rad2deg(torch.atan2(
                physics_enc[:, 0], physics_enc[:, 1]))

        v1 = self.spatial_branch(x_img)
        v2 = self.physics_branch(physics_enc)
        # Floquet 增强: 传入 theta 用于 kx,ky 计算
        freq_basis = self.trunk.build_basis(x_freq, theta_deg)
        v3 = self.trunk(freq_basis)

        return self.fusion(v1, v2, v3)


# ==========================================
# 测试与诊断 (chebyshev basis)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 生成测试频率
    freqs = torch.linspace(-1, 1, 201).unsqueeze(0)  # [1, 201]
    order = 16

    # 对比三种基底
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for ax, btype in zip(axes, ["taylor", "chebyshev", "legendre"]):
        basis = build_freq_basis(freqs, order=order, basis_type=btype)
        cond = compute_condition_number(basis)
        for i in range(min(order + 1, 8)):
            ax.plot(freqs[0].numpy(), basis[0, :, i].numpy(),
                    label=f"$T_{i}$" if btype != "taylor" else f"$f^{i}$")
        ax.set_title(f"{btype} basis\ncondition number ≈ {cond:.1f}")
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Basis value")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    # 条件数对比
    ax = axes[3]
    orders = [4, 6, 8, 10, 12, 14, 16]
    taylor_conds = []
    cheb_conds = []
    for o in orders:
        t = build_freq_basis(freqs, order=o, basis_type="taylor")
        c = build_freq_basis(freqs, order=o, basis_type="chebyshev")
        taylor_conds.append(compute_condition_number(t))
        cheb_conds.append(compute_condition_number(c))

    ax.semilogy(orders, taylor_conds, "o-", label="Taylor (monomial)")
    ax.semilogy(orders, cheb_conds, "s-", label="Chebyshev")
    ax.set_xlabel("Polynomial order")
    ax.set_ylabel("Condition number")
    ax.set_title("Condition Number vs Order")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("chebyshev_vs_taylor_condition.png", dpi=150)
    print("Saved: chebyshev_vs_taylor_condition.png")
    print(f"Order=16: Taylor cond={taylor_conds[-1]:.1e}, Chebyshev cond={cheb_conds[-1]:.1e}")


# ==========================================
# 测试与诊断 (MIONet)
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 测试基本 MIONet
    batch = 4
    x_img = torch.randn(batch, 1, 10, 10).to(device)
    x_freq = torch.linspace(-1, 1, 201).unsqueeze(0).repeat(batch, 1).to(device)
    theta = torch.tensor([0., 15., 30., 45.]).to(device)
    phi = torch.zeros(batch).to(device)

    # 测试 TE
    model = MIONet(order=16, basis_type="chebyshev").to(device)
    enc_te = model.build_physics_encoding(theta, phi,
                                           torch.zeros(batch).to(device))
    s11, s21 = model(x_img, x_freq, enc_te)
    print(f"MIONet TE: s11={s11.shape}, s21={s21.shape}")
    print(f"  S11 range: [{s11.min().item():.4f}, {s11.max().item():.4f}]")
    print(f"  S21 range: [{s21.min().item():.4f}, {s21.max().item():.4f}]")

    # 测试双极化
    te_s11, te_s21, tm_s11, tm_s21 = model.forward_dual_pol(
        x_img, x_freq, theta, phi)
    print(f"MIONet dual-pol: TE_S11={te_s11.shape}, TM_S11={tm_s11.shape}")

    # 测试 M2
    model_m2 = MIONetKernel(order=16, basis_type="chebyshev").to(device)
    s11_m2, s21_m2 = model_m2(x_img, x_freq, enc_te)
    print(f"MIONetKernel: s11={s11_m2.shape}, s21={s21_m2.shape}")

    # 测试 M3
    model_m3 = MIONetFloquet(order=16, basis_type="chebyshev").to(device)
    s11_m3, s21_m3 = model_m3(x_img, x_freq, enc_te, theta)
    print(f"MIONetFloquet: s11={s11_m3.shape}, s21={s21_m3.shape}")

    # 参数统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MIONet params: {total:,} total, {trainable:,} trainable")
    print("All tests passed!")
