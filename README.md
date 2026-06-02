---
# StreamVAD 一键部署工具 & StreamVAD-XL 改进版本
基于 [StreamVAD](https://github.com/Han-lijun/StreamVAD) 的课程作业部署脚本，自动完成环境配置、依赖安装与代码生成，实现本地摄像头或视频文件的实时异常检测。  
同时提供 **StreamVAD-XL** 增强版本，引入 Transformer-XL 长期记忆机制，强化流式视频异常检测框架。
---
## 功能特性
### StreamVAD
- **一键部署**：自动克隆仓库、安装全部依赖、生成推理文件
- **实时 HUD 显示**：画面叠加异常评分、颜色渐变进度条、实时帧率（FPS）、推理延迟（ms）
- **异常警报**：分数超过阈值时画面变红边框并弹出提示文字
- **会话统计摘要**：退出后在终端打印完整统计报告，给出整体异常判定
- **双模式输入**：支持本地摄像头与视频文件，自动同步播放帧率
### StreamVAD-XL 模型改进版本
- **Transformer-XL 长期记忆机制**：替代原始独立块级时序建模，缓存循环记忆复用历史隐藏状态，解决原始 StreamVAD 缺失长距离时序上下文的问题
- **分离记忆梯度截断**：对历史记忆停止梯度回传，避免梯度爆炸，降低长视频 GPU 显存消耗
- **多尺度与长期联合建模**：保留原始 MTS（多尺度时序选择）与 KCG（关键片段生成器）模块，MTS 捕获局部突变异常，Transformer-XL 记忆建模全局缓变异常
- **全流式训练管线**：逐块增量训练，持续更新整个视频的记忆缓存，符合真实监控流式场景
- **多目标损失函数**：在原始 MIL 损失基础上增加时序平滑损失、稀疏损失和 KCG 正则化损失
- **Warm-up + Cosine Annealing 学习率**：解决 Transformer 模块训练振荡，提升收敛精度
- **梯度裁剪 & 特征归一化**：防止梯度爆炸，平衡特征分布

---
## 环境要求
| 项目 | 要求 |
|------|------|
| Python | 3.8 – 3.10（推荐 3.9） |
| Git | 已安装并在 PATH 中 |
| 硬件 | CPU 或 CUDA GPU|
| 操作系统 | Windows |

> ```bash
> conda create -n streamvad python=3.9
> conda activate streamvad
> ```
---
## 依赖版本一览
| 包名 | 版本 |
|------|------|
| torch | 2.2.0 |
| torchvision | 0.17.0 |
| torchaudio | 2.2.0 |
| numpy | 1.24.3 |
| opencv-python | 4.8.0.76 |
| pillow | 10.0.0 |
| scikit-learn | 1.3.0 |
| scipy | 1.10.1 |
| clip (OpenAI) | latest |
---
## 快速部署（基础版推理）
### 第一步：运行部署脚本
将 `deploy_streamvad.py` 下载到本地，在终端执行：
```bash
python deploy_streamvad.py
```
脚本将自动完成以下 7 个步骤：
```
[1/7] 检查 Python 版本
[2/7] 检查 Git 安装
[3/7] 克隆 StreamVAD 仓库
[4/7] 安装全部依赖（PyTorch / CLIP 等）
[5/7] 生成实时推理文件
[6/7] 预下载 CLIP ViT-B/32 模型（约 350 MB）
[7/7] 部署完成，打印后续步骤提示
```
### 第二步：放置预训练权重
前往官方仓库下载 `xd_best.pth`：
```
https://github.com/Han-lijun/StreamVAD#setup
```
将文件放入自动创建的目录：
```
StreamVAD/
└── weights/
    └── xd_best.pth   ← 放在这里
```
### 第三步：运行检测
```bash
cd StreamVAD
python realtime_inference.py
```
按 **`q`** 退出，终端将打印会话统计摘要。
---
## 运行模式
### 模式一：本地摄像头（默认）
无需额外修改，脚本默认使用设备编号 `0`（第一个摄像头）。
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
右上角显示运行设备和已运行时长（`MM:SS`）。
**颜色含义：**
| 颜色 | 含义 |
|------|------|
| 🟢 绿色 | 分数 < 阈值 × 0.75，正常 |
| 🟡 黄色 | 分数介于 0.75 × 阈值 ~ 阈值之间，注意 |
| 🔴 红色 | 分数 ≥ 阈值，检测到异常 |
检测到异常时，画面四周出现红色粗边框，底部居中显示 `ANOMALY DETECTED!`。
---
## 会话统计摘要
按 `q` 退出后，终端自动打印如下格式的报告：
```
============================================================
           StreamVAD 会话统计摘要
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
**整体判定规则：**
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
| `args.anomaly_threshold` | `0.5` | 异常报警阈值，范围 0–1 |
| `VIDEO_SOURCE` | `0` | 视频源，`0` = 本地摄像头，字符串 = 文件路径 |
| `FRAME_INTERVAL` | `1` | 每隔多少帧取一帧 |
| `args.attn_window` | `16` | 每次推理使用的帧数 |
| `args.cache_keep_max_len` | `64` | 模型内部时序缓存长度 |
---
## 生成的文件结构
部署完成后，`StreamVAD/` 目录下新增以下文件：
```
StreamVAD/
├── realtime_input.py       # 视频流读取与 CLIP 预处理
├── clip_extractor.py       # CLIP 特征提取封装
├── realtime_inference.py   # 主推理循环
└── weights/
    └── xd_best.pth         # 预训练权重
```
---
## StreamVAD-XL 训练命令
### UCF-Crime
```bash
python train_xl.py --dataset ucf \
  --train_list list\ucf_CLIP_rgb.csv \
  --batch_size 8 \
  --mem_len 32 \
  --chunk_size 16 \
  --epochs 50
```
### 关键训练参数说明
| 参数 | 说明 |
|------|------|
| `--dataset` | 数据集名称，如 `ucf` |
| `--train_list` | 训练列表 CSV 路径 |
| `--batch_size` | 批大小 |
| `--mem_len` | Transformer-XL 记忆长度 |
| `--chunk_size` | 每个视频块大小 |
| `--epochs` | 训练轮数 |
---
## 参考
- 原始模型仓库：[https://github.com/Han-lijun/StreamVAD](https://github.com/Han-lijun/StreamVAD)
- OpenAI CLIP：[https://github.com/openai/CLIP](https://github.com/openai/CLIP)
