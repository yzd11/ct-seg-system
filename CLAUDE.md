# CLAUDE.md — CT 肝脏与肿瘤分割系统

本科毕业设计项目：面向CT影像的深度学习肿瘤分割与定量分析辅助诊断平台。

## 项目概述

- **论文题目**：面向CT影像的深度学习肿瘤分割与定量分析辅助诊断平台的实现
- **作者**：禹治东，郑州轻工业大学 软件学院
- **数据集**：LiTS2017（131 例腹部增强 CT，512×512 轴向切片）
- **任务**：3 类语义分割——背景(0)、肝脏(1)、肿瘤(2)
- **模型**：4 种 2D U-Net 变体（U-Net、ResU-Net、U-Net++、Attention U-Net++）
- **系统**：Vue3 + FastAPI + Celery + Redis + Docker Compose
- **论文**：`UNet-LiTS2017-main/542213330145-禹治东-毕设论文.docx`

## 目录结构

```
ct-seg-system/
├── README.md                   # 项目说明与快速启动
├── CLAUDE.md                   # 本文件（开发者文档）
├── docker-compose.yml          # 4 容器编排（redis + backend + worker + frontend）
├── .env.example                # 环境变量模板
├── .gitignore
│
├── backend/                    # FastAPI 后端服务
│   ├── app/
│   │   ├── main.py             # FastAPI 入口 + CORS + 路由挂载
│   │   ├── config.py           # pydantic-settings 配置
│   │   ├── database.py         # 异步 SQLAlchemy + 自动迁移
│   │   ├── models/             # ORM: Case, InferenceJob, SliceResult
│   │   ├── schemas/            # Pydantic: CaseCreate/Response, JobCreate/Response
│   │   ├── routers/            # cases, nifti, inference, export 四个路由
│   │   ├── services/           # nifti_service, inference_service, model_registry,
│   │   │                       # overlay_service, export_service
│   │   ├── tasks/              # Celery 配置 + run_inference 异步任务
│   │   ├── ml_models/          # 推理用模型架构（4 个模型，从训练代码适配）
│   │   └── utils/              # CT 窗宽窗位、体积计算
│   ├── weights/                # .pth 权重文件（不入 git）
│   ├── uploads/                # 上传的 NIfTI + 推理结果（不入 git）
│   └── ct_seg.db               # SQLite 运行时数据库
│
├── frontend/                   # Vue3 前端 SPA
│   ├── src/
│   │   ├── App.vue             # 根布局：AppHeader + AppSidebar + router-view
│   │   ├── router/index.js     # 5 个路由（Home, Upload, Viewer, Compare, History）
│   │   ├── api/                # Axios 封装（client, cases, inference, nifti）
│   │   ├── stores/             # Pinia 状态（caseStore, jobStore, viewerStore）
│   │   ├── views/              # 5 个页面视图
│   │   ├── components/         # 11 个组件
│   │   │   ├── common/         # PageHeader
│   │   │   ├── layout/         # AppHeader, AppSidebar
│   │   │   ├── viewer/         # SliceCanvas, SliceSlider, OverlayControls, WindowingControls
│   │   │   ├── inference/      # ModelSelector, ProgressBar, MetricsPanel
│   │   │   └── export/         # ExportButton
│   │   └── styles/theme.css    # 完整设计系统
│   └── nginx.conf              # SPA 回退 + /api/ 反向代理
│
├── training/                   # 模型训练代码（论文实验）
│   ├── train.py                # 训练入口（YAML 配置 / CLI 参数双模式）
│   ├── test.py                 # 测试/评估入口
│   ├── dataset.py              # CustomImageDataset：PNG 切片 → tensor
│   ├── evaluate_nnunet.py      # nnU-Net 对比评估
│   ├── models/                 # 4 种 U-Net 架构（论文最终版本）
│   │   ├── __init__.py         # MODEL_REGISTRY: unet, resunet, unet_pp, attention_unet_pp
│   │   ├── unet.py             # 标准 U-Net（DoubleConv + 跳跃连接）
│   │   ├── resunet.py          # ResU-Net（ResBlock 替代 DoubleConv）
│   │   ├── unet_pp.py          # U-Net++（嵌套密集跳跃连接 + 深监督）
│   │   ├── attention_unet_pp.py # Attention U-Net++（密集连接 + 20 个 AttentionGate）
│   │   └── attention_unet.py   # AttentionGate 类（被 attention_unet_pp 依赖）
│   ├── training_lib/           # 训练基础设施
│   │   ├── trainer.py          # Trainer：AMP + 早停 + checkpoint + 日志
│   │   └── augmentations.py    # 4 级数据增强（none/basic/medium/strong）
│   ├── utils/                  # losses.py（CombinedLoss）+ metrics.py（Dice/IoU/HD95）
│   ├── experiments/
│   │   ├── configs/            # YAML 实验配置 + ExperimentConfig dataclass
│   │   └── tracking/           # MLflow 日志（无 MLflow 时降级 CSV）
│   ├── data/                   # 5 折分层交叉验证划分
│   ├── preprocess/             # NIfTI→PNG 预处理脚本
│   ├── scripts/                # 分析/评估/绘图脚本
│   ├── logs/                   # 完整训练日志
│   ├── assets/                 # 训练曲线 SVG
│   └── CLAUDE.md               # 训练方法论文档
│
└── _archived/                  # 旧版本文件（待确认后删除）
    ├── README.md               # 归档说明
    ├── attention_unet.py       # 独立 Attention U-Net（论文未使用）
    └── trainer_root_copy.py    # server 根目录的重复 trainer.py
```

## 论文实验结果

| 模型 | 参数量 | Liver Dice | Tumor Dice | Tumor HD95 | 推理速度 |
|------|--------|:----------:|:----------:|:----------:|:--------:|
| U-Net | 17.26M | 0.8893 | 0.7815 | 19.24 px | 12.69 ms |
| ResU-Net | **14.18M** | 0.8863 | 0.7749 | 18.66 px | **11.28 ms** |
| U-Net++ | 24.82M | 0.8867 | 0.7817 | **15.54 px** | 25.23 ms |
| Att. U-Net++ | 25.60M | **0.8918** | **0.8000** (5-fold) | 16.47 px | 26.91 ms |

**关键结论**：
- Att. U-Net++ Dice 最优（验证集 0.8928/0.8000），肿瘤 Recall 最高（0.634）
- U-Net++ 边界精度最优（HD95=15.54px，ASSD=3.88px）
- ResU-Net 性价比最优（14.18M 参数，88.6 FPS）
- 消融实验验证：tumor_weight=3.0 + CE+Dice 组合损失是此任务的最优策略

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/cases/` | 上传 NIfTI（multipart，≤600MB） |
| GET | `/api/v1/cases/` | 案例列表 |
| DELETE | `/api/v1/cases/{id}` | 删除案例及关联数据 |
| GET | `/api/v1/nifti/{id}/metadata` | NIfTI 元数据（切片数、体素间距） |
| GET | `/api/v1/nifti/{id}/slice/{idx}?center=50&width=400` | CT 切片灰度 PNG |
| POST | `/api/v1/inference/jobs` | 提交推理任务 `{case_id, model_name}` |
| GET | `/api/v1/inference/jobs/{id}` | 任务状态/进度 |
| GET | `/api/v1/inference/jobs/{id}/results` | 逐切片面积/周长数据 |
| GET | `/api/v1/inference/jobs/{id}/mask/{idx}` | 分割蒙版 RGBA PNG |
| DELETE | `/api/v1/inference/jobs/{id}` | 取消/删除推理任务 |
| GET | `/api/v1/inference/cases/{id}/jobs` | 某案例的所有任务 |
| POST | `/api/v1/export/pdf?job_id=` | 导出中文 PDF 诊断报告 |

## 训练快速参考

```bash
cd training/

# 训练（需在 training/ 目录下运行，或设置 PYTHONPATH）
python train.py --config experiments/configs/unet_baseline.yaml --fold_id 0

# 单模型 5 折交叉验证
for fold in 0 1 2 3 4; do
    python train.py --config experiments/configs/unet_baseline.yaml --fold_id $fold
done

# 评估
python test.py --model unet --data_root /path/to/LiTS

# 批量评估（4 模型 × 5 折）
python scripts/evaluate_all_folds.py

# 消融实验
bash scripts/run_ablation.sh
```

## 关键约定（贯穿训练与推理系统）

1. **标签映射**：PNG 像素 `0=背景, 128→1=肝脏, 255→2=肿瘤`，训练时通过 `label_mapping` 转为 `0/1/2`
2. **RGB 3 通道**：灰度 CT 经 `Image.open().convert("RGB")` 复制到 3 通道，所有模型 `in_channels=3`
3. **CT 窗宽窗位**：`center=50, width=400`（腹部软组织窗），默认 HU 区间 [-150, 250]
4. **图像尺寸**：统一 `256×256`，BILINEAR 插值
5. **随机种子**：3407（全局，`cudnn.deterministic=True`）
6. **深监督**：U-Net++ / Att. U-Net++ 训练时返回 4 个 tensor 的列表（loss 取平均），eval 时返回单个 tensor
7. **训练配置**：Adam(lr=0.001)，StepLR(step=20, gamma=0.1)，CombinedLoss(0.5×CE+0.5×Dice)，tumor_weight=3.0，batch_size=8，AMP 默认开启，早停 patience=15

## 本地开发启动

```bash
# 0. 确保 Redis 运行
redis-cli ping  # → PONG

# 1. 后端
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# 2. Celery Worker（新终端，Windows 必须 --pool=solo）
cd backend
.venv\Scripts\activate
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo

# 3. 前端（新终端）
cd frontend
npm run dev

# 访问 http://localhost:5173
# API 文档 http://localhost:8000/docs
```

## Docker 部署

```bash
docker-compose up --build
# 前端 → http://localhost
# API 文档 → http://localhost:8000/docs
```

## 注意事项

- **权重文件**（`.pth`）和 **上传数据**（`uploads/`）不入 Git
- Celery worker `--concurrency=1`（GPU 显存限制）
- 模型 LRU 缓存最多 2 个（`MAX_CACHED_MODELS=2`）
- 推理系统 `backend/app/ml_models/` 和训练代码 `training/models/` 是两套独立的模型代码副本
- 推理系统仅部署 4 个模型（无独立 Attention U-Net），与论文一致
- 本系统仅用于学术研究，**不适用于临床诊断**
