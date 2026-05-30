# models/attention_unet.py
# AttentionGate 模块：被 Attention U-Net++ (attention_unet_pp.py) 复用
#
# 注意力门原理：
#   给定门控信号 g（来自深层解码器，语义更强）和跳跃连接特征 x（来自编码器同层），
#   学习一个空间注意力图 alpha ∈ [0,1]，对 x 做逐像素加权，抑制背景噪声。
#
# 参考论文：Oktay et al., "Attention U-Net: Learning Where to Look for
#           the Pancreas", MIDL 2018.

import torch
import torch.nn as nn
import torch.nn.functional as F


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
