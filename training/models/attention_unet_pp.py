# models/attention_unet_pp.py
# Attention U-Net++ 实现（含深度监督）
#
# 核心创新（相对于 U-Net++）：
#   在每个密集节点 X[i][j] 的 cat 操作之前，对所有前序同层节点 X[i][0..j-1]
#   各施加一个独立的 AttentionGate，门控信号统一使用上采样后的下层特征 g=Up(X[i+1][j-1])。
#   注意力门过滤无关背景后，再与 g 一起 cat 进入 DoubleConv。
#
# 与各模型的对比：
#   U-Net       : 直接 cat(skip, up)，无过滤，无密集连接
#   Attention U-Net : 有注意力，但 skip 路径单一
#   U-Net++     : 密集连接，但 skip 无过滤
#   Attention U-Net++: 密集连接 + 每条 skip 都过注意力门（本模型）
#
# 节点公式（来自论文 li2020）：
#   X[i][j] = Φ( Ag(X[i,0],g), Ag(X[i,1],g), ..., Ag(X[i,j-1],g), g )
#   其中 g = Up(X[i+1][j-1])，Ag 为 AttentionGate
#
# 通道配置：
#   注意力门不改变通道数，因此每个节点的 in_ch 与 U-Net++ 完全相同：
#   in_ch = j × nb_filter[i] + nb_filter[i+1]
#
# 深度监督：
#   训练时返回 [out1, out2, out3, out4]（对应 X[0][1]~X[0][4]），
#   推理时仅返回 out4，test.py 零改动。
#
# 参考论文：Chen Li et al., "Attention UNet++: A Nested Attention-Aware
#           U-Net for Liver CT Image Segmentation", ICIP 2020.

import torch
import torch.nn as nn

from models.unet import DoubleConv, Down, OutConv
from models.attention_unet import AttentionGate
from models.unet_pp import _upsample_like


class AttentionUNetPP(nn.Module):
    """
    Attention U-Net++

    编码器与 U-Net++ 完全相同（nb_filter=[64,128,256,512,512]）。
    密集节点的 DoubleConv 输入通道与 U-Net++ 相同（注意力门不改变通道数）。
    每个密集节点额外引入 j 个 AttentionGate（j = 前序节点数），门控信号为上采样特征。

    注意力门数量统计：
      level3: X[3][1]=1                          共 1 个
      level2: X[2][1]=1, X[2][2]=2               共 3 个
      level1: X[1][1]=1, X[1][2]=2, X[1][3]=3   共 6 个
      level0: X[0][1]=1, X[0][2]=2, X[0][3]=3, X[0][4]=4  共 10 个
      总计：20 个 AttentionGate
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 3,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(AttentionUNetPP, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        nb = [base_c, base_c*2, base_c*4, base_c*8, base_c*8]
        # nb = [64, 128, 256, 512, 512]

        # ── 编码器（与 U-Net/U-Net++ 完全相同）──────────────────────────────
        self.conv0_0 = DoubleConv(in_channels, nb[0])
        self.down1   = Down(nb[0], nb[1])
        self.down2   = Down(nb[1], nb[2])
        self.down3   = Down(nb[2], nb[3])
        self.down4   = Down(nb[3], nb[4])

        # ── 密集节点 DoubleConv（通道配置与 U-Net++ 完全一致）───────────────
        # level 3
        self.conv3_1 = DoubleConv(1*nb[3] + nb[4], nb[3])   # 1024→512

        # level 2
        self.conv2_1 = DoubleConv(1*nb[2] + nb[3], nb[2])   # 768→256
        self.conv2_2 = DoubleConv(2*nb[2] + nb[3], nb[2])   # 1024→256

        # level 1
        self.conv1_1 = DoubleConv(1*nb[1] + nb[2], nb[1])   # 384→128
        self.conv1_2 = DoubleConv(2*nb[1] + nb[2], nb[1])   # 512→128
        self.conv1_3 = DoubleConv(3*nb[1] + nb[2], nb[1])   # 640→128

        # level 0
        self.conv0_1 = DoubleConv(1*nb[0] + nb[1], nb[0])   # 192→64
        self.conv0_2 = DoubleConv(2*nb[0] + nb[1], nb[0])   # 256→64
        self.conv0_3 = DoubleConv(3*nb[0] + nb[1], nb[0])   # 320→64
        self.conv0_4 = DoubleConv(4*nb[0] + nb[1], nb[0])   # 384→64

        # ── 注意力门（每个密集节点独立一组 ModuleList）──────────────────────
        # X[3][1]：1个AG，门控信号=Up(x4_0)[512]，skip=x3_0[512]
        self.ag3_1 = nn.ModuleList([
            AttentionGate(nb[4], nb[3], nb[3]//2),           # F_g=512,F_l=512,F_int=256
        ])

        # X[2][1]：1个AG，门控信号=Up(x3_0)[512]，skip=x2_0[256]
        self.ag2_1 = nn.ModuleList([
            AttentionGate(nb[3], nb[2], nb[2]//2),           # F_g=512,F_l=256,F_int=128
        ])
        # X[2][2]：2个AG，门控信号=Up(x3_1)[512]，skip=x2_0/x2_1[256]
        self.ag2_2 = nn.ModuleList([
            AttentionGate(nb[3], nb[2], nb[2]//2),
            AttentionGate(nb[3], nb[2], nb[2]//2),
        ])

        # X[1][1]：1个AG，门控信号=Up(x2_0)[256]，skip=x1_0[128]
        self.ag1_1 = nn.ModuleList([
            AttentionGate(nb[2], nb[1], nb[1]//2),           # F_g=256,F_l=128,F_int=64
        ])
        # X[1][2]：2个AG，门控信号=Up(x2_1)[256]，skip=x1_0/x1_1[128]
        self.ag1_2 = nn.ModuleList([
            AttentionGate(nb[2], nb[1], nb[1]//2),
            AttentionGate(nb[2], nb[1], nb[1]//2),
        ])
        # X[1][3]：3个AG，门控信号=Up(x2_2)[256]，skip=x1_0/x1_1/x1_2[128]
        self.ag1_3 = nn.ModuleList([
            AttentionGate(nb[2], nb[1], nb[1]//2),
            AttentionGate(nb[2], nb[1], nb[1]//2),
            AttentionGate(nb[2], nb[1], nb[1]//2),
        ])

        # X[0][1]：1个AG，门控信号=Up(x1_0)[128]，skip=x0_0[64]
        self.ag0_1 = nn.ModuleList([
            AttentionGate(nb[1], nb[0], nb[0]//2),           # F_g=128,F_l=64,F_int=32
        ])
        # X[0][2]：2个AG，门控信号=Up(x1_1)[128]，skip=x0_0/x0_1[64]
        self.ag0_2 = nn.ModuleList([
            AttentionGate(nb[1], nb[0], nb[0]//2),
            AttentionGate(nb[1], nb[0], nb[0]//2),
        ])
        # X[0][3]：3个AG，门控信号=Up(x1_2)[128]，skip=x0_0/x0_1/x0_2[64]
        self.ag0_3 = nn.ModuleList([
            AttentionGate(nb[1], nb[0], nb[0]//2),
            AttentionGate(nb[1], nb[0], nb[0]//2),
            AttentionGate(nb[1], nb[0], nb[0]//2),
        ])
        # X[0][4]：4个AG，门控信号=Up(x1_3)[128]，skip=x0_0/x0_1/x0_2/x0_3[64]
        self.ag0_4 = nn.ModuleList([
            AttentionGate(nb[1], nb[0], nb[0]//2),
            AttentionGate(nb[1], nb[0], nb[0]//2),
            AttentionGate(nb[1], nb[0], nb[0]//2),
            AttentionGate(nb[1], nb[0], nb[0]//2),
        ])

        # ── 深度监督输出头（与 U-Net++ 完全相同）────────────────────────────
        self.out1 = OutConv(nb[0], num_classes)   # X[0][1]
        self.out2 = OutConv(nb[0], num_classes)   # X[0][2]
        self.out3 = OutConv(nb[0], num_classes)   # X[0][3]
        self.out4 = OutConv(nb[0], num_classes)   # X[0][4] 最终输出

    def forward(self, x: torch.Tensor):
        """
        训练时返回 list[logits×4]，推理时返回单个 logits tensor。
        输入：(B, in_channels, H, W)
        输出：(B, num_classes, H, W) 或 [(B, num_classes, H, W) × 4]
        """
        # ── 编码器 ────────────────────────────────────────────────────────────
        x0_0 = self.conv0_0(x)     # (B, 64,  H,    W)
        x1_0 = self.down1(x0_0)    # (B, 128, H/2,  W/2)
        x2_0 = self.down2(x1_0)    # (B, 256, H/4,  W/4)
        x3_0 = self.down3(x2_0)    # (B, 512, H/8,  W/8)
        x4_0 = self.down4(x3_0)    # (B, 512, H/16, W/16)

        # ── Level 3 ───────────────────────────────────────────────────────────
        g3_1 = _upsample_like(x4_0, x3_0)                   # (B, 512, H/8)
        x3_1 = self.conv3_1(torch.cat([
            self.ag3_1[0](g3_1, x3_0),                       # attended x3_0
            g3_1,
        ], dim=1))                                            # (B, 512, H/8)

        # ── Level 2 ───────────────────────────────────────────────────────────
        g2_1 = _upsample_like(x3_0, x2_0)                   # (B, 512, H/4)
        x2_1 = self.conv2_1(torch.cat([
            self.ag2_1[0](g2_1, x2_0),
            g2_1,
        ], dim=1))                                            # (B, 256, H/4)

        g2_2 = _upsample_like(x3_1, x2_0)                   # (B, 512, H/4)
        x2_2 = self.conv2_2(torch.cat([
            self.ag2_2[0](g2_2, x2_0),
            self.ag2_2[1](g2_2, x2_1),
            g2_2,
        ], dim=1))                                            # (B, 256, H/4)

        # ── Level 1 ───────────────────────────────────────────────────────────
        g1_1 = _upsample_like(x2_0, x1_0)                   # (B, 256, H/2)
        x1_1 = self.conv1_1(torch.cat([
            self.ag1_1[0](g1_1, x1_0),
            g1_1,
        ], dim=1))                                            # (B, 128, H/2)

        g1_2 = _upsample_like(x2_1, x1_0)                   # (B, 256, H/2)
        x1_2 = self.conv1_2(torch.cat([
            self.ag1_2[0](g1_2, x1_0),
            self.ag1_2[1](g1_2, x1_1),
            g1_2,
        ], dim=1))                                            # (B, 128, H/2)

        g1_3 = _upsample_like(x2_2, x1_0)                   # (B, 256, H/2)
        x1_3 = self.conv1_3(torch.cat([
            self.ag1_3[0](g1_3, x1_0),
            self.ag1_3[1](g1_3, x1_1),
            self.ag1_3[2](g1_3, x1_2),
            g1_3,
        ], dim=1))                                            # (B, 128, H/2)

        # ── Level 0 ───────────────────────────────────────────────────────────
        g0_1 = _upsample_like(x1_0, x0_0)                   # (B, 128, H)
        x0_1 = self.conv0_1(torch.cat([
            self.ag0_1[0](g0_1, x0_0),
            g0_1,
        ], dim=1))                                            # (B, 64, H)

        g0_2 = _upsample_like(x1_1, x0_0)                   # (B, 128, H)
        x0_2 = self.conv0_2(torch.cat([
            self.ag0_2[0](g0_2, x0_0),
            self.ag0_2[1](g0_2, x0_1),
            g0_2,
        ], dim=1))                                            # (B, 64, H)

        g0_3 = _upsample_like(x1_2, x0_0)                   # (B, 128, H)
        x0_3 = self.conv0_3(torch.cat([
            self.ag0_3[0](g0_3, x0_0),
            self.ag0_3[1](g0_3, x0_1),
            self.ag0_3[2](g0_3, x0_2),
            g0_3,
        ], dim=1))                                            # (B, 64, H)

        g0_4 = _upsample_like(x1_3, x0_0)                   # (B, 128, H)
        x0_4 = self.conv0_4(torch.cat([
            self.ag0_4[0](g0_4, x0_0),
            self.ag0_4[1](g0_4, x0_1),
            self.ag0_4[2](g0_4, x0_2),
            self.ag0_4[3](g0_4, x0_3),
            g0_4,
        ], dim=1))                                            # (B, 64, H)

        # ── 输出 ──────────────────────────────────────────────────────────────
        if self.training:
            return [self.out1(x0_1), self.out2(x0_2), self.out3(x0_3), self.out4(x0_4)]
        else:
            return self.out4(x0_4)


if __name__ == '__main__':
    from thop import profile

    model = AttentionUNetPP(in_channels=3, num_classes=3)
    input_tensor = torch.randn(1, 3, 256, 256)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Output shape : {output.shape}")   # 期望 (1, 3, 256, 256)

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs        : {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters   : {params / 1e6:.2f} M")

    model.train()
    outputs = model(input_tensor)
    print(f"Deep supervision: {len(outputs)} heads")
    for i, o in enumerate(outputs):
        print(f"  out{i+1}: {o.shape}")
