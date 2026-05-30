# _archived — 旧版本/重复文件

以下文件为合并过程中识别出的旧版本或重复文件，保留在此目录供确认。
确认无误后可安全删除整个 `_archived/` 目录。

## 文件说明

| 文件 | 来源 | 归档原因 |
|------|------|---------|
| `attention_unet.py` | UNet-LiTS2017-server/models/ | 独立 Attention U-Net 模型。论文最终版本未使用此模型（仅使用 Attention U-Net++）。AttentionGate 类仍被 attention_unet_pp.py 依赖，已保留在 training/models/attention_unet.py 中。 |
| `trainer_root_copy.py` | UNet-LiTS2017-server/trainer.py | server 根目录的 trainer.py，与 training/trainer.py 完全重复。正式版本在 training_lib/trainer.py。 |

## 其他注意

- `UNet-LiTS2017-main/` 整体为早期开发版本，其核心代码已被 server 版本覆盖
- `UNet-LiTS2017-server/` 中的 `logs/`, `assets/`, `mlruns/` 等运行时产物已迁移到 training/ 目录
- `ml_models/` 在 ct-seg-system/backend/app/ 中是推理系统的模型代码，与 training/models/ 为独立副本，两者都需要保留
