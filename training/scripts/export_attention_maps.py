"""
导出 Attention U-Net++ 注意力图 — 供论文章节 5.5 使用

用法:
  python scripts/export_attention_maps.py

输出:
  results/attention_maps/
    volume-XX_slice-YYY_ct.png       — 原始 CT 切片
    volume-XX_slice-YYY_gt.png       — Ground Truth 标注
    volume-XX_slice-YYY_pred.png     — 模型预测分割
    volume-XX_slice-YYY_att.png      — 注意力权重热力图(α 叠加 CT)

原理:
  对每个 AttentionGate,其 psi 模块输出 1 通道 Sigmoid 后的 α∈[0,1]。
  forward hook 在 psi 之后、逐元素乘 x 之前截获 α 值。
  选择 level 0 的 ag0_4[0] (最浅层的第一个 skip) — 分辨率为 256×256,
  对应编码器最浅层特征 x0_0,空间细节最丰富。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms

from models.attention_unet_pp import AttentionUNetPP
from dataset import label_mapping


# ── 配置 ────────────────────────────────────────────────────
CHECKPOINT = 'models/attention_unet_pp/best_fold1_seed3407.pth'
DATA_ROOT = 'H:\\LiTS'               # 按实际路径修改
OUT_DIR   = 'results/attention_maps'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE  = 256

# 代表性切片: (case_id, slice_index) — 选有肝脏+肿瘤的切片
CANDIDATES = [
    ('volume-12', 120),     # fold 0, tumor Dice=0.956 — 分割质量最好的 case
    ('volume-129', 150),    # fold 0, tumor Dice=0.891 — 中等难度
    ('volume-8', 100),      # fold 1, tumor Dice=0.849 — 另一折的代表
]

os.makedirs(OUT_DIR, exist_ok=True)


# ── 加载模型 ────────────────────────────────────────────────
print(f'Loading {CHECKPOINT} ...')
model = AttentionUNetPP(in_channels=3, num_classes=3)
state = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print('Model loaded.')


# ── 注册 hook 捕获注意力 α ──────────────────────────────────
_alpha_maps = {}

def _hook_factory(name):
    """每个 AG 注册一个 hook,在 psi 的 Sigmoid 之后捕获 α"""
    def hook(module, input, output):
        # output = psi(x) = Sigmoid(BN(Conv1×1(ReLU(W_g+W_x))))
        # shape: (B, 1, H, W), values ∈ [0, 1]
        _alpha_maps[name] = output.detach().cpu()
    return hook

# 在 level 0 的注意力门上注册 hook(分辨率最高 256×256)
hooks = []
for idx, ag in enumerate(model.ag0_4):
    # ag.psi 是 Sequential(Conv2d, BN, Sigmoid)
    h = ag.psi.register_forward_hook(_hook_factory(f'ag0_4_{idx}'))
    hooks.append(h)
# 也注册 ag0_1(最浅层, 仅 1 个 skip)
for idx, ag in enumerate(model.ag0_1):
    h = ag.psi.register_forward_hook(_hook_factory(f'ag0_1_{idx}'))
    hooks.append(h)

print(f'Registered {len(hooks)} hooks.')


# ── 推理并保存 ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE),
                      interpolation=transforms.InterpolationMode.NEAREST),
])

cmap = plt.cm.jet
norm = mcolors.Normalize(vmin=0, vmax=1)

for case_id, slice_idx in CANDIDATES:
    img_dir = os.path.join(DATA_ROOT, case_id, 'Image')
    gt_dir  = os.path.join(DATA_ROOT, case_id, 'GT')
    if not os.path.exists(img_dir):
        print(f'{case_id}: SKIP (no images)'); continue

    fnames = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
    if slice_idx >= len(fnames):
        slice_idx = len(fnames) // 2   # fallback to middle slice

    fname = fnames[slice_idx]

    # ── 加载图像 ───────────────────────────────────────────
    image = Image.open(os.path.join(img_dir, fname)).convert('RGB')
    label = Image.open(os.path.join(gt_dir, fname)).convert('L')

    image = transform(image)
    label = transform(label)
    label_np = np.array(label)
    label_np = np.vectorize(label_mapping.get)(label_np)  # {0,128,255}→{0,1,2}

    image_t = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)

    # ── 前向推理 ───────────────────────────────────────────
    _alpha_maps.clear()
    with torch.no_grad():
        output = model(image_t)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

    # ── 保存 CT ────────────────────────────────────────────
    ct_np = np.array(image)     # (256, 256, 3) RGB
    Image.fromarray(ct_np).save(
        os.path.join(OUT_DIR, f'{case_id}_slice{slice_idx:03d}_ct.png'))

    # ── 保存 GT ────────────────────────────────────────────
    gt_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    gt_rgb[label_np == 1] = [0, 200, 0]    # 肝脏绿色
    gt_rgb[label_np == 2] = [220, 30, 30]  # 肿瘤红色
    Image.fromarray(gt_rgb).save(
        os.path.join(OUT_DIR, f'{case_id}_slice{slice_idx:03d}_gt.png'))

    # ── 保存预测 ───────────────────────────────────────────
    pred_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    pred_rgb[pred == 1] = [0, 200, 0]
    pred_rgb[pred == 2] = [220, 30, 30]
    Image.fromarray(pred_rgb).save(
        os.path.join(OUT_DIR, f'{case_id}_slice{slice_idx:03d}_pred.png'))

    # ── 保存注意力热力图(选 ag0_4[0]: 最浅层 skip x0_0 的 α) ─
    if 'ag0_4_0' in _alpha_maps:
        alpha = _alpha_maps['ag0_4_0'][0, 0].numpy()  # (256, 256)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(ct_np, cmap='gray')
        im = ax.imshow(alpha, cmap='jet', alpha=0.55, vmin=0, vmax=1)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Attention α')
        plt.tight_layout(pad=0)
        fig.savefig(os.path.join(OUT_DIR, f'{case_id}_slice{slice_idx:03d}_att.png'),
                    dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f'{case_id} slice {slice_idx}: α mean={alpha.mean():.3f}, '
              f'max={alpha.max():.3f}')
    else:
        print(f'{case_id}: no alpha captured')

# 清理
for h in hooks:
    h.remove()

print(f'\nDone. Results in: {OUT_DIR}')
print('Use files: *_ct.png, *_gt.png, *_pred.png, *_att.png')
# 推荐插入论文: 用 volume-12 的 _att.png 作为图 5-3
# 图注: "Attention U-Net++ level-0 注意力门(ag0_4)的空间权重热力图,
#        叠加于原始 CT 切片。暖色区域(α→1)为模型重点关注区域,
#        冷色区域(α→0)为被抑制的背景区域。"
