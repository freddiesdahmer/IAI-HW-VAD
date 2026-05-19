# IAI-HW-VAD
# StreamVAD 实时视频异常检测
## 项目简介
本项目是 StreamVAD 视频异常检测模型的一键部署工具，专为课程作业设计，可快速搭建环境并实现本地摄像头实时异常检测，自动完成依赖安装、代码生成和模型预下载。

## 环境要求
- Python 3.9
- 已安装 Git
- 支持 CPU 运行

## 快速部署
1. 下载 `deploy_streamvad.py` 脚本到本地
2. 打开终端，运行部署脚本：
   ```bash
   python deploy_streamvad.py
   ```
3. 下载官方预训练权重 `xd_best.pth`，放入自动生成的 `StreamVAD/weights/` 文件夹

以下是修正格式后的内容，已修复代码块、步骤编号及缩进问题：

```markdown
## 运行检测

### 方式 1：本地摄像头检测

运行以下命令：

```bash
cd StreamVAD
python realtime_inference.py
```

- 窗口会实时显示视频和异常分数
- 当分数超过 0.5 时，画面变红并提示异常
- 按 `q` 键退出程序

### 方式 2：本地视频文件检测

1. **修改 `realtime_input.py`**  
   找到 `RealtimeVideoStream` 类的 `__init__` 方法，新增一行获取视频原帧率：

   ```python
   def __init__(self, source=0, clip_length=16, frame_interval=1):
       self.source = source
       self.clip_length = clip_length
       self.frame_interval = frame_interval
       self.cap = cv2.VideoCapture(source)
       self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       self.cap.set(cv2.CAP_PROP_FPS, 30)  # 原有代码保留，新增下面这行
       self.fps = self.cap.get(cv2.CAP_PROP_FPS)
       self.frame_buffer = []
       self.running = False
   ```

2. **修改 `realtime_inference.py`**  
   找到视频源配置行，替换为你的视频文件路径：
   ```python
   VIDEO_SOURCE = "./test_video.mp4"
   ```

3. **调整播放速度**  
   找到主循环末尾的退出判断行，替换为：

   ```python
   wait_time = int(1000 / video_stream.fps) if video_stream.fps > 0 else 1
   if cv2.waitKey(wait_time) & 0xFF == ord('q'):
       break
   ```

运行命令与摄像头模式完全一致：

```bash
cd StreamVAD
python realtime_inference.py
```
```

## 简单参数调整
修改 `realtime_inference.py` 即可：
- `args.anomaly_threshold`：异常报警阈值
- `VIDEO_SOURCE`：视频源（0=本地摄像头）


## 参考
原项目：https://github.com/Han-lijun/StreamVAD
