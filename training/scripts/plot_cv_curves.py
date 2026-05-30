"""从 TensorBoard event 提取 5 折验证 Dice，按模型聚合 mean±std 并出图"""
import os
import sys
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS_DIR = sys.argv[1] if len(sys.argv) > 1 else 'runs'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'results/curves'
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ['unet', 'resunet', 'unet_pp', 'attention_unet_pp']
COLORS = {
    'unet': '#3A5A7C',
    'resunet': '#7AA2C5',
    'unet_pp': '#D4A574',
    'attention_unet_pp': '#C0392B',
}
LABELS = {
    'unet': 'U-Net',
    'resunet': 'ResU-Net',
    'unet_pp': 'U-Net++',
    'attention_unet_pp': 'Att U-Net++',
}
METRICS = ['dice_liver_valid', 'dice_tumor_valid']
TITLES = ['Dice Liver (Validation)', 'Dice Tumor (Validation)']
YLIMS = [(0.84, 0.92), (0.72, 0.86)]  # 各自聚焦数据范围，放大差异
SMOOTH_WINDOW = 3  # 移动平均窗口，epoch 数

# 提取数据: {model: {metric: {epoch: [fold1, fold2, ...]}}}
data = {m: {metric: defaultdict(list) for metric in METRICS} for m in MODELS}

for model in MODELS:
    for fold in range(5):
        logdir = os.path.join(RUNS_DIR, model, f'fold{fold}_seed3407')
        if not os.path.isdir(logdir):
            print(f'  SKIP: {logdir}')
            continue
        try:
            ea = EventAccumulator(logdir)
            ea.Reload()
            for metric in METRICS:
                if metric in ea.Tags().get('scalars', []):
                    for event in ea.Scalars(metric):
                        epoch = event.step
                        value = event.value
                        data[model][metric][epoch].append(value)
            print(f'  {model} fold{fold}: {len(ea.Tags().get("scalars", []))} tags')
        except Exception as e:
            print(f'  {model} fold{fold}: ERROR - {e}')

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax_idx, (metric, title) in enumerate(zip(METRICS, TITLES)):
    ax = axes[ax_idx]
    for model in MODELS:
        d = data[model][metric]
        if not d:
            continue
        epochs = sorted(d.keys())
        raw_means = [np.mean(d[e]) for e in epochs]
        raw_stds = [np.std(d[e]) for e in epochs]

        # 移动平均平滑
        w = SMOOTH_WINDOW
        means = np.convolve(raw_means, np.ones(w)/w, mode='valid')
        stds = np.convolve(raw_stds, np.ones(w)/w, mode='valid')
        smooth_epochs = epochs[w-1:]

        line = ax.plot(smooth_epochs, means, color=COLORS[model], label=LABELS[model], linewidth=1.5)
        ax.fill_between(smooth_epochs,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=COLORS[model], alpha=0.12)
    ax.set_title(title, fontsize=13, fontweight='normal')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Dice', fontsize=11)
    ax.legend(fontsize=9, frameon=True, loc='lower right')
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(*YLIMS[ax_idx])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

plt.tight_layout(pad=2)
out_path = os.path.join(OUT_DIR, 'validation_dice_curves.svg')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
print(f'\nSaved: {out_path} (.svg + .png)')
