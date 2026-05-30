# models/attention_unet.py
# Attention U-Net 实现
#
# 核心改动（相对于标准 U-Net）：
#   在每个解码器上采样块中，skip connection 的编码器特征经过 AttentionGate 加权后再 concat，
#   抑制无关背景区域，增强肝脏/肿瘤区域的响应。
#
# 参考论文：Oktay et al., "Attention U-Net: Learning Where to Look for
#           the Pancreas", MIDL 2018.
#
# 注意力门原理：
#   给定门控信号 g（来自深层解码器，语义更强）和跳跃连接特征 x（来自编码器同层），
#   学习一个空间注意力图 alpha ∈ [0,1]，对 x 做逐像素加权，抑制背景噪声。
#
# 与标准 U-Net 的参数量对比：
#   每个注意力门新增约 2 × F_int 个卷积参数（F_int = skip通道数 // 2），总增量约 0.5-1M

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import DoubleConv, Down, OutConv   # 直接复用 U-Net 的基础模块


class AttentionGate(nn.Module):
    """
    注意力门（Attention Gate）

    输入：
      g : 门控信号，来自解码器上采样后的特征，shape (B, F_g, H, W)
          语义信息更强，用于"告知"注意力机制应关注哪些区域
      x : skip connection 特征，来自编码器同层，shape (B, F_l, H, W)
          空间信息更丰富，是被加权的对象

    输出：
      x * alpha，shape (B, F_l, H, W)
      alpha 是空间注意力图，值域 [0,1]，高值区域被保留，低值区域被抑制

    计算流程：
      1. W_g(g) + W_x(x)  → 两路特征在中间维度 F_int 对齐并相加
      2. ReLU → psi        → 生成单通道注意力 logits
      3. Sigmoid           → 归一化到 [0,1]
      4. 上采样 alpha 到与 x 相同尺寸（因 g 可能比 x 小一倍）
      5. x * alpha         → 空间加权
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        F_g  : 门控信号通道数（解码器上采样后）
        F_l  : skip 特征通道数（编码器同层）
        F_int: 中间维度（通常取 F_l // 2，控制注意力参数量）
        """
        super(AttentionGate, self).__init__()

        # 门控路径：将 g 映射到中间维度 F_int（步长为1，不改变空间尺寸）
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # skip 路径：将 x 映射到中间维度 F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # 注意力输出：F_int → 1 通道，BN + Sigmoid 生成空间注意力图
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: 门控信号（来自解码器，可能比 x 空间尺寸小）
        x: skip connection 特征（来自编码器，空间分辨率更高）
        """
        g1 = self.W_g(g)   # (B, F_int, H_g, W_g)
        x1 = self.W_x(x)   # (B, F_int, H_x, W_x)

        # 若 g 和 x 空间尺寸不一致（bilinear上采样后可能差1像素），对齐后相加
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        # 两路特征相加 → ReLU → 生成注意力 logits
        psi = self.psi(F.relu(g1 + x1, inplace=True))  # (B, 1, H_x, W_x)

        # 用注意力图对 skip 特征加权
        return x * psi


class AttentionUp(nn.Module):
    """
    Attention U-Net 的解码器上采样块

    流程：
      1. 上采样 x1（来自深层）至与 x2 相同的空间尺寸
      2. 以上采样后的特征为门控信号 g，对 x2（skip 特征）做注意力加权
      3. 将加权后的 x2 与 g concat
      4. 经 DoubleConv 融合特征

    与标准 U-Net Up 的区别：
      Up 直接 cat(x2, x1_upsampled)
      AttentionUp 先对 x2 做注意力门控，再 cat(attended_x2, x1_upsampled)
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """
        in_channels : 上采样前的通道数（深层特征 + skip 特征之和）
        out_channels: DoubleConv 输出通道数
        bilinear    : 双线性上采样（True）或转置卷积（False）
        """
        super(AttentionUp, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # bilinear 模式下上采样不改变通道数，DoubleConv 中 mid = in_channels // 2
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积先将通道数减半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # 注意力门：门控信号通道 = 上采样后通道，skip 通道 = in_channels // 2
        # F_int 取 skip 通道的一半，控制注意力参数量
        F_g  = in_channels // 2        # 上采样后（bilinear 不改通道）或 ConvTranspose 后
        F_l  = in_channels // 2        # skip 特征通道（编码器对应层）
        F_int = F_l // 2               # 中间维度
        self.attention = AttentionGate(F_g=F_g, F_l=F_l, F_int=max(F_int, 1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: 来自下一层（深层）的特征，需要上采样，shape (B, C_deep, H/2, W/2)
        x2: 来自编码器同层的 skip connection 特征，shape (B, C_skip, H, W)
        """
        x1 = self.up(x1)

        # 尺寸对齐（处理奇数尺寸边界情况，与 U-Net 保持一致）
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 注意力门：用 x1（门控信号）对 x2（skip 特征）加权
        x2 = self.attention(g=x1, x=x2)

        # Concat 并经 DoubleConv 融合
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net

    与标准 U-Net 结构完全相同，唯一区别是解码器中每个 Up 块替换为 AttentionUp，
    即 skip connection 传入前经过注意力门过滤。

    编码器/Bottleneck/输出层与 UNet 完全一致，可直接对比参数量差异。

    参数：
      in_channels  : 输入通道数（默认 3，灰度图 RGB 复制）
      num_classes  : 输出类别数（默认 3：背景/肝脏/肿瘤）
      bilinear     : 上采样方式（默认 True，双线性插值）
      base_c       : 基础通道数（默认 64）
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 3,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # ── 编码器（与 UNet 完全相同）────────────────────────────────────────
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1   = Down(base_c,     base_c * 2)
        self.down2   = Down(base_c * 2, base_c * 4)
        self.down3   = Down(base_c * 4, base_c * 8)
        self.down4   = Down(base_c * 8, base_c * 16 // factor)   # bottleneck

        # ── 解码器（Up → AttentionUp，其余通道配置不变）─────────────────────
        self.up1 = AttentionUp(base_c * 16,     base_c * 8 // factor, bilinear)
        self.up2 = AttentionUp(base_c * 8,      base_c * 4 // factor, bilinear)
        self.up3 = AttentionUp(base_c * 4,      base_c * 2 // factor, bilinear)
        self.up4 = AttentionUp(base_c * 2,      base_c,               bilinear)

        # ── 输出层（与 UNet 完全相同）────────────────────────────────────────
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，接口与 UNet.forward 完全一致。
        输入：(B, in_channels, H, W)
        输出：(B, num_classes, H, W) logits
        """
        # 编码器
        x1 = self.in_conv(x)    # (B, 64,  H,    W)
        x2 = self.down1(x1)     # (B, 128, H/2,  W/2)
        x3 = self.down2(x2)     # (B, 256, H/4,  W/4)
        x4 = self.down3(x3)     # (B, 512, H/8,  W/8)
        x5 = self.down4(x4)     # (B, 512, H/16, W/16)  bottleneck

        # 解码器（带注意力门）
        x = self.up1(x5, x4)    # (B, 256, H/8,  W/8)
        x = self.up2(x,  x3)    # (B, 128, H/4,  W/4)
        x = self.up3(x,  x2)    # (B, 64,  H/2,  W/2)
        x = self.up4(x,  x1)    # (B, 64,  H,    W)

        logits = self.out_conv(x)  # (B, num_classes, H, W)
        return logits


if __name__ == '__main__':
    # 验证模型结构、输出尺寸和参数量
    from thop import profile

    model = AttentionUNet(in_channels=3, num_classes=3)
    input_tensor = torch.randn(1, 3, 256, 256)

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs     : {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")

    output = model(input_tensor)
    print(f"Output shape: {output.shape}")   # 期望 (1, 3, 256, 256)
