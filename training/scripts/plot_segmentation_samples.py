"""生成图5-2：代表性测试切片分割结果可视化对比"""
import os, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────────────
DATA_ROOT = r'H:\LiTS'
TEST_TXT = r'F:\code-storage\PyCharm\UNet-LiTS2017-server\preprocess\test.txt'
MODEL_DIR = r'F:\code-storage\PyCharm\UNet-LiTS2017-server\models'
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, r'F:\code-storage\PyCharm\UNet-LiTS2017-server')
from models import MODEL_REGISTRY
from dataset import label_mapping

MODELS = ['unet', 'resunet', 'unet_pp', 'attention_unet_pp']
MODEL_LABELS = ['U-Net', 'ResU-Net', 'U-Net++', 'Att U-Net++']
COLORMAP = {0: [0, 0, 0], 1: [128, 128, 128], 2: [255, 255, 255]}  # 0=bg, 1=liver, 2=tumor

# ── 选取代表性 case+slice ─────────────────────────────────────────
# 根据之前测试结果：volume-57(tumor Dice~0.92)好, volume-21(~0.77)中, volume-4(~0.55)差, volume-70(~0.89)好
SAMPLES = [
    ('volume-57', '140'),   # easy tumor
    ('volume-70', '110'),   # good
    ('volume-21', '90'),    # medium
    ('volume-4', '80'),     # hard
]

# ── 加载所有模型 checkpoint ───────────────────────────────────────
models = {}
for model_name in MODELS:
    # 使用 fold0 的 checkpoint（选一个代表性的）
    weight_path = os.path.join(MODEL_DIR, model_name, f'best_fold0_seed3407.pth')
    if not os.path.exists(weight_path):
        weight_path = os.path.join(MODEL_DIR, model_name, 'best_fold0_seed3407.pth')
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(in_channels=3, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    models[model_name] = model
    print(f'Loaded {model_name}: {weight_path}')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
])
to_tensor = transforms.ToTensor()

# ── 推理 & 绘图 ───────────────────────────────────────────────────
n_rows = len(SAMPLES)
n_cols = 2 + len(MODELS)  # CT + GT + 4 models = 6 cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

col_titles = ['CT Image', 'Ground Truth'] + MODEL_LABELS
for col in range(n_cols):
    axes[0, col].set_title(col_titles[col], fontsize=10, fontweight='normal')

for row_idx, (case, slice_name) in enumerate(SAMPLES):
    img_path = os.path.join(DATA_ROOT, case, 'Image', f'{slice_name}.png')
    gt_path = os.path.join(DATA_ROOT, case, 'GT', f'{slice_name}.png')

    # 检查文件是否存在，不存在则尝试第一个可用切片
    if not os.path.exists(img_path):
        img_dir = os.path.join(DATA_ROOT, case, 'Image')
        fnames = sorted(os.listdir(img_dir))
        slice_name = fnames[len(fnames) // 2].replace('.png', '')
        img_path = os.path.join(DATA_ROOT, case, 'Image', f'{slice_name}.png')
        gt_path = os.path.join(DATA_ROOT, case, 'GT', f'{slice_name}.png')

    # 原图
    image = Image.open(img_path).convert('RGB')
    image_resized = transform(image)
    image_t = to_tensor(image_resized).unsqueeze(0).to(DEVICE)

    axes[row_idx, 0].imshow(np.array(image_resized), cmap='gray')
    axes[row_idx, 0].set_ylabel(f'{case}\n#{slice_name}', fontsize=8)

    # GT
    label = Image.open(gt_path).convert('L')
    label_resized = transform(label)
    label_np = np.array(label_resized)
    # 映射标签值用于可视化: 0->0, 128->1(liver gray), 255->2(tumor white)
    gt_vis = np.zeros((*label_np.shape, 3), dtype=np.uint8)
    for val, rgb in [(0, [0,0,0]), (128, [100,100,100]), (255, [255,255,255])]:
        gt_vis[label_np == val] = rgb
    axes[row_idx, 1].imshow(gt_vis)

    # 各模型预测
    for col_idx, model_name in enumerate(MODELS):
        with torch.no_grad():
            output = models[model_name](image_t)
            pred = torch.argmax(output, dim=1)[0].cpu().numpy()

        # 可视化: 0=bg(黑), 1=liver(灰), 2=tumor(白)
        pred_vis = np.zeros((*pred.shape, 3), dtype=np.uint8)
        pred_vis[pred == 1] = [100, 100, 100]
        pred_vis[pred == 2] = [255, 255, 255]
        axes[row_idx, 2 + col_idx].imshow(pred_vis)

    # 关闭坐标轴
    for col_idx in range(n_cols):
        axes[row_idx, col_idx].axis('off')

plt.tight_layout(pad=0.5)
out_dir = r'F:\code-storage\PyCharm\UNet-LiTS2017-main\results\figures'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'segmentation_samples.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'\nSaved: {out_path}')
