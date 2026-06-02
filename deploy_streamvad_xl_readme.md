# StreamVAD-XL 快速部署指南

## 快速部署（基础版推理）

### 第一步：运行部署脚本

将 `deploy_streamvad_xl.py` 下载到本地，在终端执行：

```bash
python deploy_streamvad_xl.py
```

脚本将自动完成以下 7 个步骤：

| 步骤 | 说明 |
|------|------|
| [1/7] | 检查 Python 版本（要求 3.8.0 – 3.10.12） |
| [2/7] | 检查 Git 安装 |
| [3/7] | 创建 `StreamVAD_XL` 项目目录 |
| [4/7] | 安装全部依赖（PyTorch / CLIP 等） |
| [5/7] | 生成实时推理文件 |
| [6/7] | 预下载 CLIP ViT-B/32 模型（约 350 MB） |
| [7/7] | 部署完成，打印后续步骤提示 |

> **注意**：步骤 [3/7] 仅创建项目目录，XL 模型代码（`model.py`、`dataset.py`）需另行下载放入，详见第二步。

### 第二步：放置模型代码与权重

#### 2.1 放置模型代码

将 XL 模型的 `model.py`（包含 `StreamVADWithXL` 类）和 `dataset.py` 复制到项目目录：

```
StreamVAD_XL/
├── model.py          ← 包含 StreamVADWithXL、EnhancedPCI、TransformerXLMemoryBlock
└── dataset.py        ← 数据集定义（推理阶段非必须，但建议保留）
```

#### 2.2 放置预训练权重

将 XL 模型权重文件 `xl_best.pth` 放入自动创建的目录：

```
StreamVAD_XL/
└── weights/
    └── xl_best.pth   ← 放在这里
```

### 第三步：运行检测

```bash
cd StreamVAD_XL
python realtime_inference.py
```

按 **q** 退出，终端将打印会话统计摘要。

---

## 运行模式

### 模式一：本地摄像头（默认）

无需额外修改，脚本默认使用设备编号 0（第一个摄像头）。

### 模式二：本地视频文件

编辑 `realtime_inference.py`，将视频源改为文件路径：

```python
VIDEO_SOURCE = "./test_video.mp4"
```

同时将主循环末尾的退出判断替换为帧率同步版本：

```python
wait_time = int(1000 / video_stream.fps) if video_stream.fps > 0 else 1
if cv2.waitKey(wait_time) & 0xFF == ord('q'):
    break
```

---

## 画面 HUD 说明

运行时检测窗口左上角会显示半透明信息面板：

```
┌─────────────────────────────────────┐
│ Score : 0.3821                      │
│ [██████████░░░░░░░░░░░│░░░░░░░░░░] │  ← 颜色进度条（│ 为阈值位置）
│ Threshold: 0.50                     │
│ FPS :  29.8                         │
│ Latency:  42.3 ms                   │
└─────────────────────────────────────┘
```

右上角显示运行设备和已运行时长（MM:SS）。

### 颜色含义

| 颜色 | 含义 |
|------|------|
| 🟢 绿色 | 分数 < 阈值 × 0.75，正常 |
| 🟡 黄色 | 分数介于 0.75 × 阈值 ~ 阈值之间，注意 |
| 🔴 红色 | 分数 ≥ 阈值，检测到异常 |

检测到异常时，画面四周出现红色粗边框，底部居中显示 **ANOMALY DETECTED!**。

---

## 会话统计摘要

按 **q** 退出后，终端自动打印如下格式的报告：

```
============================================================
           StreamVAD-XL 会话统计摘要
============================================================
  运行时长       : 45.2 秒
  处理帧数       : 1356
  平均帧率       : 30.0 FPS
  推理片段数     : 84
  异常片段数     : 12  (14.3%)
  平均异常分数   : 0.3821
  最高异常分数   : 0.7634
  最低异常分数   : 0.1022
  异常阈值       : 0.50

  整体判定       : [中风险] 视频中出现间歇性异常，请注意关键片段。
============================================================
```

### 整体判定规则

| 异常片段占比 | 判定等级 |
|-------------|---------|
| < 10% | 正常 |
| 10% – 30% | 中风险 |
| ≥ 30% | 高风险 |

---

## 参数调整

所有关键参数均在 `realtime_inference.py` 的 `main()` 函数顶部：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `feature_dim` | 512 | CLIP 特征维度（ViT-B/32 输出为 512，一般无需修改） |
| `hidden_dim` | 256 | Transformer-XL 隐层维度 |
| `mem_len` | 32 | 记忆缓存最大长度，越大则时序上下文越远，显存占用越高 |
| `num_layers` | 2 | Transformer-XL 层数 |
| `nhead` | 4 | 多头注意力头数 |
| `chunk_size` | 16 | 每次推理使用的帧数（需与 `RealtimeVideoStream` 的 `clip_length` 一致） |
| `anomaly_threshold` | 0.5 | 异常报警阈值，范围 0–1 |
| `VIDEO_SOURCE` | 0 | 视频源，0 = 本地摄像头，字符串 = 文件路径 |
| `FRAME_INTERVAL` | 1 | 每隔多少帧取一帧 |

> **提示**：`feature_dim`、`hidden_dim`、`mem_len`、`num_layers`、`nhead` 为模型结构参数，必须与训练时一致，否则无法加载权重。`anomaly_threshold`、`VIDEO_SOURCE`、`FRAME_INTERVAL` 可根据实际场景自由调整。

---

## 与原版 StreamVAD 的区别

| 项目 | StreamVAD（原版） | StreamVAD-XL |
|------|-------------------|--------------|
| 模型类 | `LSHMA` | `StreamVADWithXL` |
| 时序建模 | 固定窗口注意力 | Transformer-XL 记忆缓存（跨片段传递） |
| 特征处理 | 直接输入 | L2 归一化后输入 |
| 权重文件 | `xd_best.pth` | `xl_best.pth` |
| 前向接口 | `model(visual, flag, lengths, ...)` | `model(chunk, memories)` |
| 记忆管理 | 无 | 自动维护 `memories` 状态 |

---

## 生成的文件结构

部署完成后，`StreamVAD_XL/` 目录下包含以下文件：

```
StreamVAD_XL/
├── model.py                # XL 模型定义（需手动放入）
├── dataset.py              # 数据集定义（需手动放入）
├── realtime_input.py       # 视频流读取与 CLIP 预处理
├── clip_extractor.py       # CLIP 特征提取封装
├── realtime_inference.py   # 主推理循环
└── weights/
    └── xl_best.pth         # XL 预训练权重（需手动放入）
```

---

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.8.0 – 3.10.12 |
| PyTorch | 2.2.0（CUDA 11.8） |
| torchvision | 0.17.0 |
| CLIP | openai/CLIP（GitHub） |
| OpenCV | 4.8.0.76 |
| NumPy | 1.24.3 |
| scikit-learn | 1.3.0 |

推荐使用 conda 创建环境：

```bash
conda create -n streamvad_xl python=3.9
conda activate streamvad_xl
```
