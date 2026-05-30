# 毕业设计：基于 LiTS2017 的肝脏与肿瘤分割

## 项目背景

基于 LiTS2017 数据集，逐步构建并对比四种 2D 分割网络，最终以 nnU-Net (3D) 作为上限参考基线。

**目标任务**：CT 图像中的肝脏（类别1）与肿瘤（类别2）分割，共3类（含背景）。

---

## 一、数据集

- **原始数据**：`volumes/volume-0.nii` 到 `volume-130.nii`（131个样本），`segmentations/segmentation-0.nii` 到 `segmentation-130.nii`
- **服务器路径**：`/root/autodl-tmp/volumes/` 和 `/root/autodl-tmp/segmentations/`
- **标签映射**：0=背景，1=肝脏，2=肿瘤
- **预处理流程**：
  - `preprocess/split_nii.py`：NIfTI 按 z 轴切片成 2D PNG，只保留有标签的切片，4线程
  - CT 调窗：窗宽=400，窗位=50（腹部软组织窗）
  - 标签值重映射：1→128，2→255（存储为 uint8 PNG）
  - 输出结构：`/root/autodl-tmp/LiTS/{case_id}/Image/` 和 `.../GT/`
  - `preprocess/split_dataset.py`：生成 train/valid/test 的 txt 文件（需在 preprocess/ 目录下执行）
- **数据划分**（seed=3407）：train=78，valid=26，test=27，共 131 个 case
- **切片数量**：train≈11519，valid≈3498

---

## 二、环境配置

### 服务器（AutoDL）
- 镜像：PyTorch 2.x + CUDA 11.8 + Python 3.10
- conda 环境名：`U-Net`

### 依赖安装
```bash
conda activate U-Net
pip install opencv-python nibabel thop matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard==2.14.0 setuptools==69.5.1 protobuf==3.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 三、代码结构

```
UNet-LiTS2017-main/
├── models/
│   ├── __init__.py           # 模型注册表，MODEL_REGISTRY
│   ├── unet.py               # ① U-Net（已完成）
│   ├── attention_unet.py     # ② Attention U-Net（已完成）
│   ├── resunet.py            # ③ ResU-Net（已完成）
│   ├── unet_pp.py            # ④ 待实现
│   ├── attention_unet_pp.py  # ⑤ 待实现
│   └── unet/best.pth         # 训练后自动生成
├── utils/
│   ├── losses.py             # DiceLoss + CombinedLoss（CE+Dice 各50%，支持类别加权）
│   └── metrics.py            # calculate_metrics：Dice/IoU；calculate_hd95_single：HD95
├── preprocess/
│   ├── split_nii.py          # 路径已配置为服务器路径，4线程
│   ├── split_dataset.py      # 路径已配置为服务器路径
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
├── dataset.py                # CustomImageDataset，data_root 参数化
├── train.py                  # 通用训练脚本，--model 参数选择模型
├── test.py                   # 通用测试脚本
├── model.py                  # 原始文件保留（可忽略）
├── runs/{model_name}/        # TensorBoard 日志，按模型名区分
├── results/{model_name}/     # 测试结果（score.csv + summary.csv + 可视化图）
└── CLAUDE.md
```

---

## 四、统一训练配置（所有模型必须相同）

| 参数 | 值 | 说明 |
|------|:--:|------|
| in_channels | 3 | RGB 复制灰度图 |
| num_classes | 3 | 背景/肝脏/肿瘤 |
| img_size | 256 | 512 会导致肿瘤过拟合 |
| batch_size | 8 | batch=16 会减少肿瘤迭代次数 |
| epochs | 30 | - |
| lr | 0.001 | Adam 初始学习率 |
| weight_decay | 0 | weight_decay 会压制肿瘤特征 |
| scheduler | StepLR(step=20, gamma=0.1) | epoch 1-20 lr=0.001，21-30 lr=0.0001 |
| tumor_weight | 3.0 | CE+Dice 同时对肿瘤加权 |
| 保存策略 | 0.4×Dice_liver + 0.6×Dice_tumor | 肿瘤权重更高 |
| seed | 3407 | - |
| num_workers | 4 | 服务器，Windows 本地用 0 |

> **调参教训**：
> - `img_size=512`：肝脏改善但肿瘤严重过拟合，已放弃
> - `batch=16`：减少每轮迭代次数，伤害肿瘤学习，已放弃
> - `CosineAnnealingLR`：无骤降巩固阶段，肿瘤特征不稳定，已放弃
> - `weight_decay=1e-4`：压制肿瘤大幅度权重特征，已放弃

---

## 五、服务器运行命令

### 训练
```bash
cd /root/UNet-LiTS2017-main
conda activate U-Net

nohup python train.py \
    --model unet \
    --tumor_weight 3.0 \
    --data_root /root/autodl-tmp/LiTS \
    --num_workers 4 > train.log 2>&1 &

tail -f train.log
```

### TensorBoard
```bash
# 关闭已有进程
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9

# 启动（端口 6007，AutoDL 控制台映射）
tensorboard --port 6007 --logdir runs
```

### 测试
```bash
python test.py \
    --model unet \
    --data_root /root/autodl-tmp/LiTS \
    --tumor_weight 3.0
```

---

## 六、六阶段实施计划

### ① 2D U-Net —— ✅ 已完成

**测试结果**：Dice_liver=0.8893，Dice_tumor=0.7815，IoU_liver=0.8324，IoU_tumor=0.7368，HD95_liver=21.63px，HD95_tumor=17.23px，推理速度=12.69ms/img（78.8 FPS），参数量=17.26M

---

### ② Attention U-Net —— ✅ 已完成

**测试结果**：Dice_liver=0.8867，Dice_tumor=0.7766，IoU_liver=0.8309，IoU_tumor=0.7337，HD95_liver=22.68px，HD95_tumor=18.79px，推理速度=11.16ms/img（89.6 FPS），参数量=17.61M

**核心改动**：在 skip connection 处插入 Attention Gate，用解码器特征（门控信号 g）对编码器特征（x）做空间注意力加权，抑制无关背景区域。

**新增文件**：`models/attention_unet.py`

**模块设计**：
```
AttentionGate(F_g, F_l, F_int):
    W_g  : Conv2d(F_g, F_int, 1) + BN
    W_x  : Conv2d(F_l, F_int, 1) + BN
    psi  : Conv2d(F_int, 1, 1) + BN + Sigmoid
    forward: alpha = psi(ReLU(W_g(g) + W_x(x)))
             return x * alpha

AttentionUp: 上采样 → AttentionGate(g=上采样结果, x=skip) → cat → DoubleConv
```

**复用**：`dataset.py`、`losses.py`、`metrics.py`、`train.py` 完全不变。

---

### ③ ResU-Net —— ✅ 已完成

**核心改动**：将编码器和解码器中的 DoubleConv 替换为残差块（ResBlock），输入通过 1×1 卷积对齐通道数后与两层卷积的输出相加，引入跳跃连接缓解梯度消失。

**新增文件**：`models/resunet.py`

**模块设计**：
```
ResBlock(in_channels, out_channels):
    shortcut : Conv2d(in_channels, out_channels, 1) + BN   # 通道对齐
    主路径   : Conv3×3 + BN + ReLU → Conv3×3 + BN
    forward  : ReLU(主路径(x) + shortcut(x))

ResDown : MaxPool2d(2) → ResBlock
ResUp   : Upsample → cat(skip) → ResBlock
```

**测试结果**：Dice_liver=0.8863，Dice_tumor=0.7749，IoU_liver=0.8289，IoU_tumor=0.7305，HD95_liver=23.74px，HD95_tumor=18.55px，推理速度=11.28ms/img（88.6 FPS），参数量=14.18M

**消融结论**：参数量比 U-Net 减少 18%（14.18M vs 17.26M），精度略低于 U-Net，说明在此任务上单纯残差连接不能提升分割精度，但显著提升参数效率。HD95_tumor（18.55）优于 Attention U-Net（18.79），边界定位有一定优势。

**复用**：`train.py`/`dataset.py`/工具函数完全复用。

---

### ④ U-Net++ —— ✅ 已完成

**核心改动**：在编码器和解码器之间插入嵌套密集节点 `X[i][j]`，每个节点聚合来自同一深度所有前序节点的输出，丰富跨尺度特征融合路径。额外启用深度监督：训练时对 X[0][1]~X[0][4] 四路输出取均值 loss，加速浅层收敛。

**新增文件**：`models/unet_pp.py`

**节点逻辑**：
```
X[i][0]：原始编码器节点
X[i][j] = DoubleConv(cat(X[i][0], ..., X[i][j-1], UpConv(X[i+1][j-1])))
最终输出：X[0][4]
in_ch 公式：j × nb_filter[i] + nb_filter[i+1]
```

**测试结果**：Dice_liver=0.8867，Dice_tumor=0.7817，IoU_liver=0.8301，IoU_tumor=0.7374，HD95_liver=21.12px，HD95_tumor=15.86px，推理速度=25.23ms/img（39.6 FPS），参数量=24.82M

**结果分析**：Dice/IoU 与 U-Net 基本持平，但 HD95_tumor=15.86px 是目前四个模型最优，比 U-Net 改善约 8%，说明密集连接显著提升了肿瘤边界定位精度。推理速度较慢（参数量最多）。

**复用**：`dataset.py`/工具函数完全复用；`train.py` 新增 4 行深度监督兼容逻辑（isinstance 判断），对其他模型零影响。

---

### ⑤ Attention U-Net++（核心创新）—— ✅ 已完成

**核心改动**：在 U-Net++ 的每一个密集 skip 节点处，对来自前序节点的特征均施加注意力门，再进行拼接。共 20 个 AttentionGate，深度监督同 U-Net++。

**新增文件**：`models/attention_unet_pp.py`

**设计要点**：
```
X[i][j] 的计算：
  g        = Up(X[i+1][j-1])
  attended = [AttentionGate(F_g=nb[i+1], F_l=nb[i])(g, X[i][k]) for k in range(j)]
  X[i][j]  = DoubleConv(cat(attended + [g]))
  in_ch 与 U-Net++ 相同（注意力门不改变通道数）
```

**测试结果**：Dice_liver=0.8918，Dice_tumor=0.7808，IoU_liver=0.8369，IoU_tumor=0.7367，HD95_liver=20.34px，HD95_tumor=17.79px，推理速度=30.68ms/img（32.6 FPS），参数量≈25.6M

**结果分析**：
- 肝脏三项指标（Dice/IoU/HD95）全面最优，注意力机制对大目标聚焦效果显著
- 肿瘤 HD95=17.79px，不如 U-Net++（15.86px）：密集连接对肿瘤边界的贡献被注意力门部分抵消
- 说明注意力机制和密集连接对不同目标的作用机制不同，是有价值的消融发现

**参考论文**：Chen Li et al., "Attention UNet++", ICIP 2020（与本项目直接对应）

---

### ⑥ nnU-Net (3D) —— 待实现（独立流程）

**与①-④完全独立**，使用官方 nnU-Net v2 框架。

**主要工作**：
1. 数据格式转换：
   - `volumes/volume-X.nii` → `Dataset001_LiTS/imagesTr/liver_X_0000.nii.gz`
   - `segmentations/segmentation-X.nii` → `Dataset001_LiTS/labelsTr/liver_X.nii.gz`
   - 生成 `dataset.json`
2. 运行三条命令：
```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
nnUNetv2_train 001 3d_fullres 0
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres
```

---

## 七、对比实验结果

| 模型 | Dice_liver | Dice_tumor | IoU_liver | IoU_tumor | HD95_liver | HD95_tumor | 速度(ms) | 参数量 | 状态 |
|------|:----------:|:----------:|:---------:|:---------:|:----------:|:----------:|:--------:|:------:|:----:|
| ① U-Net | 0.8893 | 0.7815 | 0.8324 | 0.7368 | 21.63px | 17.23px | 12.69 | 17.26M | ✅ 完成 |
| ② Attention U-Net | 0.8867 | 0.7766 | 0.8309 | 0.7337 | 22.68px | 18.79px | 11.16 | 17.61M | ✅ 完成 |
| ③ ResU-Net | 0.8863 | 0.7749 | 0.8289 | 0.7305 | 23.74px | 18.55px | 11.28 | 14.18M | ✅ 完成 |
| ④ U-Net++ | 0.8867 | 0.7817 | 0.8301 | 0.7374 | 21.12px | 15.86px | 25.23 | 24.82M | ✅ 完成 |
| ⑤ Att. U-Net++ | **0.8918** | 0.7808 | **0.8369** | 0.7367 | **20.34px** | 17.79px | 30.68 | ≈25.6M | ✅ 完成 |
| ⑥ nnU-Net (3D) | - | - | - | - | - | - | - | - | 待实现 |

---

## 八、关键注意事项

1. **评估公平性**：所有模型使用同一 test set、同一 `metrics.py`、同一超参数配置
2. **类别不平衡**：肿瘤像素约占 1%，tumor_weight=3.0 对 CE+Dice 同时加权
3. **模型切换**：新增模型只需在 `models/__init__.py` 取消注释，`train.py`/`test.py` 无需改动
4. **实施顺序**：严格按 ①→②→③→④→⑤ 推进，每步与前一步对比，⑥ 单独做
5. **in_channels=3**：CT 灰度图复制为 RGB，所有模型统一，不改动
