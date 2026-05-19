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

## 运行检测
```bash
cd StreamVAD
python realtime_inference.py
```
- 窗口会实时显示视频和异常分数
- 分数超过0.5时画面变红并提示异常
- 按 `q` 键退出程序

## 简单参数调整
修改 `realtime_inference.py` 即可：
- `args.anomaly_threshold`：异常报警阈值（默认0.5）
- `VIDEO_SOURCE`：视频源（0=本地摄像头）


## 参考
原项目：https://github.com/Han-lijun/StreamVAD
