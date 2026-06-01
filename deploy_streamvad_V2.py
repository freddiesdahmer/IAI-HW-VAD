# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

REPO_URL = "https://github.com/Han-lijun/StreamVAD.git"
REPO_NAME = "StreamVAD"
PYTHON_REQUIRED = (3, 8, 0)
PYTHON_MAX = (3, 10, 12)

BASE_REQUIREMENTS = [
    "torch==2.2.0",
    "torchvision==0.17.0",
    "torchaudio==2.2.0",
    "numpy==1.24.3",
    "scikit-learn==1.3.0",
    "scipy==1.10.1",
    "tqdm==4.65.0",
    "pandas==2.0.3",
    "matplotlib==3.7.2"
]

REALTIME_REQUIREMENTS = [
    "opencv-python==4.8.0.76",
    "pillow==10.0.0",
    "ftfy==6.1.1",
    "regex==2023.8.8"
]

REALTIME_INPUT_CONTENT = '''import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

CLIP_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

class RealtimeVideoStream:
    def __init__(self, source=0, clip_length=16, frame_interval=1):
        self.source = source
        self.clip_length = clip_length
        self.frame_interval = frame_interval
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获取视频源原始帧率（用于文件播放速度同步）
        self.frame_buffer = []
        self.running = False

    def start(self):
        self.running = True
        return self

    def read_clip(self):
        self.frame_buffer.clear()
        while len(self.frame_buffer) < self.clip_length and self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            if len(self.frame_buffer) % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = CLIP_PREPROCESS(frame_pil)
                self.frame_buffer.append(frame_tensor)
        
        if len(self.frame_buffer) == self.clip_length:
            return torch.stack(self.frame_buffer).unsqueeze(0)
        return None

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
'''

CLIP_EXTRACTOR_CONTENT = '''import clip
import torch

class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_clip_feature(self, clip_tensor):
        batch_size, clip_len, c, h, w = clip_tensor.shape
        frames = clip_tensor.view(-1, c, h, w).to(self.device)
        features = self.model.encode_image(frames)
        features = features.view(batch_size, clip_len, -1)
        return features.cpu().numpy()
'''

REALTIME_INFERENCE_CONTENT = '''import torch
import cv2
import numpy as np
import threading
import time
import sys
from collections import deque
from queue import Queue
from model import LSHMA as StreamVAD
from xd_option import parser
from realtime_input import RealtimeVideoStream
from clip_extractor import CLIPFeatureExtractor


# ─────────────────────────────────────────────
#  辅助：在帧上绘制半透明矩形（用于 HUD 背景）
# ─────────────────────────────────────────────
def draw_overlay_rect(frame, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.5):
    # 在 frame 上绘制半透明填充矩形。
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ─────────────────────────────────────────────
#  辅助：绘制评分进度条
#  score 范围 [0,1]，颜色随分值从绿→黄→红渐变
# ─────────────────────────────────────────────
def draw_score_bar(frame, score, threshold, x, y, width=200, height=14):
    # 绘制带颜色渐变的评分进度条。
    # 背景槽
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # 填充宽度
    fill_w = int(np.clip(score, 0.0, 1.0) * width)
    # 颜色：低分绿色，超阈值红色，中间黄色过渡
    if score < threshold * 0.75:
        bar_color = (50, 220, 80)       # 绿
    elif score < threshold:
        bar_color = (30, 200, 240)      # 黄（BGR）
    else:
        bar_color = (40, 60, 220)       # 红
    cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)
    # 阈值刻度线
    thresh_x = x + int(threshold * width)
    cv2.line(frame, (thresh_x, y - 2), (thresh_x, y + height + 2), (255, 255, 255), 1)
    # 外框
    cv2.rectangle(frame, (x, y), (x + width, y + height), (150, 150, 150), 1)


# ─────────────────────────────────────────────
#  推理工作线程
#  新增：将入队时间戳随 feature 一起传递，用于延迟计算
# ─────────────────────────────────────────────
def inference_worker(model, feature_queue, result_queue, device, args):
    model.eval()
    video_name = torch.tensor([0], device=device)

    with torch.no_grad():
        while True:
            item = feature_queue.get()
            if item is None:
                break
            clip_feature, enqueue_time = item          # 解包特征 + 入队时间戳

            input_feature = torch.from_numpy(clip_feature).to(device)
            batch_size, clip_len, _ = input_feature.shape

            logits = model(
                visual=input_feature,
                flag='Test',
                lengths=clip_len,
                video_names=video_name,
                threshold=args.anomaly_threshold
            )

            scores = logits.squeeze().cpu().numpy()
            latency_ms = (time.time() - enqueue_time) * 1000  # 延迟（毫秒）
            result_queue.put((scores, latency_ms))


# ─────────────────────────────────────────────
#  会话结束后打印统计摘要
# ─────────────────────────────────────────────
def print_session_summary(score_history, threshold, total_frames, elapsed_sec):
    print()
    print("=" * 60)
    print("           StreamVAD 会话统计摘要")
    print("=" * 60)

    if not score_history:
        print("[INFO] 未收到任何推理结果，无统计数据。")
        print("=" * 60)
        return

    arr = np.array(score_history)
    anomaly_frames = int(np.sum(arr > threshold))
    anomaly_ratio   = anomaly_frames / len(arr) * 100
    avg_fps         = total_frames / elapsed_sec if elapsed_sec > 0 else 0.0

    print(f"  运行时长       : {elapsed_sec:.1f} 秒")
    print(f"  处理帧数       : {total_frames}")
    print(f"  平均帧率       : {avg_fps:.1f} FPS")
    print(f"  推理片段数     : {len(arr)}")
    print(f"  异常片段数     : {anomaly_frames}  ({anomaly_ratio:.1f}%)")
    print(f"  平均异常分数   : {arr.mean():.4f}")
    print(f"  最高异常分数   : {arr.max():.4f}")
    print(f"  最低异常分数   : {arr.min():.4f}")
    print(f"  异常阈值       : {threshold:.2f}")
    print()

    # 整体异常判定
    if anomaly_ratio >= 30:
        verdict = "⚠️  [高风险] 视频中存在大量异常行为，建议人工复查！"
    elif anomaly_ratio >= 10:
        verdict = "⚡ [中风险] 视频中出现间歇性异常，请注意关键片段。"
    else:
        verdict = "✅ [正常]   未检测到显著异常行为。"

    print(f"  整体判定       : {verdict}")
    print("=" * 60)


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────
def main():
    args = parser.parse_args()
    args.anomaly_threshold  = 0.5
    args.visual_width       = 512
    args.p                  = 0.1
    args.visual_layers      = 3
    args.attn_window        = 16
    args.drop_rate          = 0.5
    args.drop_attn_rate     = 0.1
    args.drop_qkv_rate      = 0.1
    args.cache_keep_max_len = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    WEIGHT_PATH    = "./weights/xd_best.pth"
    VIDEO_SOURCE   = 0
    FRAME_INTERVAL = 1

    model = StreamVAD(
        visual_width=args.visual_width,
        p=args.p,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        device=device,
        drop_rate=args.drop_rate,
        drop_attn_rate=args.drop_attn_rate,
        drop_qkv_rate=args.drop_qkv_rate,
        cache_keep_max_len=args.cache_keep_max_len
    ).to(device)

    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("[SUCCESS] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[INFO] Check weight path and ensure xd_best.pth is in weights/")
        sys.exit(1)

    feature_queue = Queue(maxsize=5)
    result_queue  = Queue(maxsize=5)

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model, feature_queue, result_queue, device, args)
    )
    inference_thread.start()
    print("[INFO] Inference thread started")

    try:
        video_stream   = RealtimeVideoStream(
            source=VIDEO_SOURCE,
            clip_length=args.attn_window,
            frame_interval=FRAME_INTERVAL
        ).start()
        clip_extractor = CLIPFeatureExtractor(device=device)
        print("[SUCCESS] Video stream and feature extractor initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize video stream: {e}")
        print("[INFO] Check camera permissions or RTSP address")
        sys.exit(1)

    # ── 统计变量 ──────────────────────────────
    current_score  = 0.0
    current_latency_ms = 0.0
    score_history  = []             # 所有片段分数，用于最终摘要

    # 帧率计算：使用滑动窗口（最近 30 帧的时间戳）
    frame_times    = deque(maxlen=30)

    # 延迟滑动平均（最近 10 次推理）
    latency_window = deque(maxlen=10)

    total_frames   = 0
    session_start  = time.time()

    print("=" * 50)
    print("StreamVAD 实时检测")
    print("按 q 键退出，会话结束后将显示统计摘要")
    print("=" * 50)

    while video_stream.running:
        # ── 读取一个 clip 并提取特征 ──────────
        clip_tensor = video_stream.read_clip()
        if clip_tensor is not None:
            clip_feature  = clip_extractor.extract_clip_feature(clip_tensor)
            enqueue_time  = time.time()
            if not feature_queue.full():
                feature_queue.put((clip_feature, enqueue_time))

        # ── 接收推理结果 ─────────────────────
        if not result_queue.empty():
            scores, lat = result_queue.get()
            current_score = float(np.mean(scores))
            score_history.append(current_score)
            latency_window.append(lat)
            current_latency_ms = float(np.mean(latency_window))
            if current_score > args.anomaly_threshold:
                print(f"⚠️  Anomaly detected! Score: {current_score:.4f}  "
                      f"Latency: {current_latency_ms:.1f} ms")

        # ── 获取原始帧用于显示 ───────────────
        ret, frame = video_stream.cap.read()
        if not ret:
            break

        total_frames += 1
        now = time.time()
        frame_times.append(now)

        # 实时帧率：用滑动窗口首尾时间差计算
        if len(frame_times) >= 2:
            display_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            display_fps = 0.0

        h_frame, w_frame = frame.shape[:2]

        # ── HUD：左上角半透明信息面板 ────────
        panel_w, panel_h = 310, 130
        draw_overlay_rect(frame, 8, 8, 8 + panel_w, 8 + panel_h,
                          color=(10, 10, 10), alpha=0.55)

        # 评分文字（颜色随分值变化）
        if current_score >= args.anomaly_threshold:
            score_color = (40, 60, 220)      # 红
        elif current_score >= args.anomaly_threshold * 0.75:
            score_color = (30, 200, 240)     # 黄
        else:
            score_color = (50, 220, 80)      # 绿

        cv2.putText(frame,
                    f"Score : {current_score:.4f}",
                    (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

        # 评分进度条
        draw_score_bar(frame, current_score, args.anomaly_threshold,
                       x=18, y=48, width=220, height=12)
        cv2.putText(frame, "0", (18, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(frame, "1", (238, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(frame,
                    f"Threshold: {args.anomaly_threshold:.2f}",
                    (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        # 帧率与延迟
        cv2.putText(frame,
                    f"FPS : {display_fps:5.1f}",
                    (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
        cv2.putText(frame,
                    f"Latency: {current_latency_ms:6.1f} ms",
                    (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)

        # 右上角：设备 & 运行时长
        elapsed = now - session_start
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        cv2.putText(frame,
                    f"Device: {device}  Time: {elapsed_str}",
                    (w_frame - 290, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # ── 异常警报覆盖层 ───────────────────
        if current_score > args.anomaly_threshold:
            # 红色边框
            cv2.rectangle(frame, (0, 0), (w_frame, h_frame), (0, 0, 220), 4)
            # 警报文字（居中偏上）
            alert_text = "ANOMALY DETECTED!"
            (tw, th), _ = cv2.getTextSize(
                alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            tx = (w_frame - tw) // 2
            cv2.putText(frame, alert_text,
                        (tx, h_frame - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 220), 2)

        cv2.imshow("StreamVAD Real-time Detection", frame)

        # 播放速度同步（视频文件时按原帧率；摄像头时最小等待 1ms）
        wait_time = max(1, int(1000 / video_stream.fps)) \
            if hasattr(video_stream, 'fps') and video_stream.fps > 0 else 1
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # ── 清理资源 ──────────────────────────────
    feature_queue.put(None)
    inference_thread.join()
    video_stream.stop()

    elapsed_total = time.time() - session_start
    print_session_summary(score_history, args.anomaly_threshold,
                          total_frames, elapsed_total)
    print("[INFO] Program exited normally")


if __name__ == "__main__":
    main()
'''


def print_banner():
    print("="*60)
    print("        StreamVAD Deployment Script v2.0")
    print("="*60)
    print()

def check_python_version():
    print("[1/7] Checking Python version...")
    current = sys.version_info
    if current < PYTHON_REQUIRED or current > PYTHON_MAX:
        print(f"[ERROR] Incompatible Python version!")
        print(f"Required: Python {PYTHON_REQUIRED[0]}.{PYTHON_REQUIRED[1]}.{PYTHON_REQUIRED[2]} - {PYTHON_MAX[0]}.{PYTHON_MAX[1]}.{PYTHON_MAX[2]}")
        print(f"Current: Python {current.major}.{current.minor}.{current.micro}")
        print("[INFO] Create environment with: conda create -n streamvad python=3.9")
        sys.exit(1)
    print(f"[SUCCESS] Python version OK: {current.major}.{current.minor}.{current.micro}")
    print()

def check_git():
    print("[2/7] Checking Git installation...")
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
        print("[SUCCESS] Git installed")
    except FileNotFoundError:
        print("[ERROR] Git not found!")
        print("[INFO] Download from: https://git-scm.com/downloads")
        sys.exit(1)
    print()

def clone_repo():
    print("[3/7] Cloning repository...")
    repo_path = Path(REPO_NAME)
    if repo_path.exists():
        print(f"[INFO] {REPO_NAME} already exists, skipping clone")
    else:
        try:
            subprocess.run(["git", "clone", REPO_URL], check=True)
            print("[SUCCESS] Repository cloned")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Clone failed: {e.stderr}")
            sys.exit(1)
    
    os.chdir(repo_path)
    print(f"[INFO] Working directory: {os.getcwd()}")
    print()

def install_dependencies():
    print("[4/7] Installing dependencies...")
    print("[INFO] Installing PyTorch (CUDA 11.8)...")
    try:
        if platform.system() in ["Windows", "Linux"]:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True)
        else:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0"
            ], check=True)
        
        print("[INFO] Installing base dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + BASE_REQUIREMENTS[3:], check=True)
        
        print("[INFO] Installing realtime dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + REALTIME_REQUIREMENTS, check=True)
        
        print("[INFO] Installing CLIP...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/openai/CLIP.git"
        ], check=True)
        
        print("[SUCCESS] All dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Installation failed: {e.stderr}")
        sys.exit(1)
    print()

def create_realtime_files():
    print("[5/7] Creating realtime inference files...")
    files = {
        "realtime_input.py": REALTIME_INPUT_CONTENT,
        "clip_extractor.py": CLIP_EXTRACTOR_CONTENT,
        "realtime_inference.py": REALTIME_INFERENCE_CONTENT
    }
    
    for filename, content in files.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[SUCCESS] Created {filename}")
    
    Path("weights").mkdir(exist_ok=True)
    print("[SUCCESS] Created weights directory")
    print()

def pre_download_clip_model():
    print("[6/7] Pre-downloading CLIP model...")
    print("[INFO] Model size ~350MB, this may take a while...")
    try:
        import clip
        clip.load("ViT-B/32", device="cpu")
        print("[SUCCESS] CLIP model downloaded")
    except Exception as e:
        print(f"[WARNING] CLIP download failed: {e}")
        print("[INFO] Will download automatically on first run")
    print()

def print_post_install():
    print("[7/7] Deployment complete!")
    print("="*60)
    print("Next steps:")
    print()
    print("1. Download pre-trained weights:")
    print("   https://github.com/Han-lijun/StreamVAD#setup")
    print("   Place xd_best.pth in StreamVAD/weights/")
    print()
    print("2. Run realtime detection:")
    print("   python realtime_inference.py")
    print()
    print("3. Press 'q' to exit — session summary will be printed")
    print("="*60)

if __name__ == "__main__":
    print_banner()
    check_python_version()
    check_git()
    clone_repo()
    install_dependencies()
    create_realtime_files()
    pre_download_clip_model()
    print_post_install()
