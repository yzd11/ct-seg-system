# CT 肝脏与肿瘤分割系统

基于 LiTS2017 数据集训练的 4 个 2D 分割模型，提供 NIfTI 文件上传、切片可视化、多模型推理对比、分割结果叠加显示与 PDF 报告导出功能。

> 本科毕业设计：面向CT影像的深度学习肿瘤分割与定量分析辅助诊断平台的实现
> 作者：禹治东，郑州轻工业大学 软件学院，2026年5月

---

## 功能概览

| 功能 | 说明 |
|------|------|
| NIfTI 上传 | 支持 `.nii` / `.nii.gz`，自动解析切片数、体素间距 |
| CT 查看器 | HTML5 Canvas 双层渲染（CT灰度 + 分割叠加），键盘/滑块切片 |
| 窗宽窗位 | 实时调节窗位/窗宽，肝脏软组织默认 C=50 W=400 |
| 模型推理 | 4 个模型异步推理，实时进度条，支持取消 |
| 多模型对比 | 最多 3 个模型并排显示同一切片的分割结果 |
| 体积估算 | 自动计算肝脏/肿瘤体积（mL），逐切片面积曲线 |
| 性能基准 | 雷达图 + 对比表格展示 4 个模型的训练测试指标 |
| 历史记录 | 查询所有历史推理任务，跳转查看结果 |
| PDF 导出 | 生成包含案例信息、体积数据和样例叠加图的报告 |

---

## 支持的模型

| 编号 | 模型 | Dice_liver | Dice_tumor | 参数量 |
|:----:|------|:----------:|:----------:|:------:|
| ① | U-Net | 0.8893 | 0.7815 | 17.26M |
| ② | ResU-Net | 0.8863 | 0.7749 | 14.18M |
| ③ | U-Net++ | 0.8867 | 0.7817 | 24.82M |
| ④ | Att. U-Net++ | **0.8918** | **0.8000** (5-fold) | 25.60M |

> 注：独立 Attention U-Net 经实验评估后未被纳入论文最终版本，仅保留 AttentionGate 模块用于 Attention U-Net++。

> 所有模型在 LiTS2017 测试集（27 个 case）上评估，输入 256×256，3 通道（灰度复制）。

---

## 技术栈

**后端**
- FastAPI + SQLAlchemy (async) + SQLite
- Celery + Redis（异步推理任务队列）
- nibabel（NIfTI 读取）
- PyTorch（模型推理）
- ReportLab（PDF 生成）

**前端**
- Vue 3 + Vite + Pinia
- Element Plus（UI 组件）
- Chart.js / vue-chartjs（雷达图、折线图）
- HTML5 Canvas（CT 切片双层渲染）

**部署**
- Docker Compose（redis + backend + celery worker + frontend/nginx）

---

## 目录结构

```
ct-seg-system/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI 入口，/api/v1
│   │   ├── config.py            # 配置（pydantic-settings）
│   │   ├── database.py          # SQLAlchemy async engine
│   │   ├── models/              # ORM 表：Case / InferenceJob / SliceResult
│   │   ├── schemas/             # Pydantic 请求/响应模型
│   │   ├── routers/             # cases / nifti / inference / export / benchmarks
│   │   ├── services/            # nifti / model_registry / inference / overlay / export
│   │   ├── tasks/               # Celery app + inference_task
│   │   └── utils/               # ct_window / metrics
│   ├── ml_models/               # 4 个分割模型（推理系统）
│   ├── weights/                 # 权重文件（不入库）
│   │   └── {model_name}/best.pth
│   └── uploads/                 # 上传的 NIfTI 文件（不入库）
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    └── src/
        ├── views/               # Home / Upload / Viewer / Compare / History
        ├── components/          # SliceCanvas / MetricsPanel / ...
        ├── stores/              # caseStore / viewerStore / jobStore
        └── api/                 # axios 封装
```

---

## 快速启动

### 方式一：Docker Compose（推荐）

**1. 放置权重文件**

```
backend/weights/
├── unet/best.pth
├── resunet/best.pth
├── unet_pp/best.pth
└── att_unet_pp/best.pth
```

**2. 启动**

```bash
docker-compose up --build
```

**3. 访问**

- 前端：http://localhost
- 后端 API 文档：http://localhost:8000/docs

---

### 方式二：本地开发（Windows）

**环境要求**：Python 3.10+，Node.js 18+，Redis

#### 0. 启动 Redis

安装 [Redis for Windows](https://github.com/tporadowski/redis/releases)，安装后作为 Windows 服务自动运行。验证：

```powershell
redis-cli ping   # 返回 PONG 即正常
```

#### 1. 放置权重文件

从训练服务器下载权重（按实际 SSH 信息替换）：

```powershell
# 先创建目录
mkdir backend\weights\unet
mkdir backend\weights\resunet
mkdir backend\weights\unet_pp
mkdir backend\weights\att_unet_pp

# 从 AutoDL 下载（按实际端口和地址替换）
scp -P <port> root@<host>:/root/UNet-LiTS2017-main/models/unet/best.pth             backend\weights\unet\best.pth
scp -P <port> root@<host>:/root/UNet-LiTS2017-main/models/resunet/best.pth          backend\weights\resunet\best.pth
scp -P <port> root@<host>:/root/UNet-LiTS2017-main/models/unet_pp/best.pth          backend\weights\unet_pp\best.pth
scp -P <port> root@<host>:/root/UNet-LiTS2017-main/models/attention_unet_pp/best.pth backend\weights\att_unet_pp\best.pth
```

#### 2. 启动后端

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

访问 http://localhost:8000/health 返回 `{"status":"ok"}` 即正常。

#### 3. 启动 Celery Worker（新开终端）

```powershell
cd backend
.venv\Scripts\activate
# Windows 必须加 --pool=solo
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo
```

看到 `celery@xxx ready.` 即正常。

#### 4. 启动前端（新开终端）

```powershell
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173

#### 各服务验证

| 服务 | 地址 | 正常标志 |
|------|------|---------|
| 后端 | http://localhost:8000/health | `{"status":"ok"}` |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| 基准数据 | http://localhost:8000/api/v1/benchmarks/ | 5条模型数据 |
| 前端 | http://localhost:5173 | 页面正常加载 |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/cases/` | 上传 NIfTI 文件 |
| GET | `/api/v1/cases/` | 案例列表 |
| GET | `/api/v1/nifti/{id}/slice/{idx}` | 获取切片 PNG（支持窗宽窗位参数） |
| POST | `/api/v1/inference/jobs` | 提交推理任务 |
| GET | `/api/v1/inference/jobs/{id}` | 查询任务状态/进度 |
| GET | `/api/v1/inference/jobs/{id}/mask/{idx}` | 获取切片分割 RGBA 叠加图 |
| GET | `/api/v1/inference/jobs/{id}/results` | 逐切片面积数据 |
| POST | `/api/v1/export/pdf?job_id=` | 导出 PDF 报告 |
| GET | `/api/v1/benchmarks/` | 4 个模型的测试集指标 |

完整文档见 `http://localhost:8000/docs`（Swagger UI）。

---

## 数据流

```
用户上传 .nii.gz
    → FastAPI 存储到 uploads/{case_id}/original.nii.gz
    → 解析 header（切片数、体素间距）写入 SQLite

选择模型 → 提交推理
    → Celery task 入队（Redis broker）
    → Worker 逐切片推理：
        nibabel 读取体积 → CT 窗化 → resize 256×256
        → model(tensor) → argmax → mask
        → 保存 RGBA PNG 到 uploads/{case_id}/results/{job_id}/masks/
        → 每10片更新进度到 SQLite
    → 前端 1.5s 轮询进度
    → 完成后体积估算写入 job 记录

查看结果
    → SliceCanvas 并行加载 CT 切片 + mask overlay
    → Canvas 合成双层，可独立控制肝脏/肿瘤显示
```

---

## 注意事项

- 权重文件（`.pth`）和上传数据均不纳入 Git 版本控制
- Celery worker `--concurrency=1`：GPU 显存有限，不并发推理
- 模型 LRU 缓存最多保留 2 个在内存中（`MAX_CACHED_MODELS=2`）
- 本项目仅用于学术研究，**不适用于临床诊断**
