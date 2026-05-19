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
import sys
from queue import Queue
from model import LSHMA as StreamVAD
from xd_option import parser
from realtime_input import RealtimeVideoStream
from clip_extractor import CLIPFeatureExtractor

def inference_worker(model, feature_queue, result_queue, device, args):
    model.eval()
    video_name = torch.tensor([0], device=device)
    
    with torch.no_grad():
        while True:
            clip_feature = feature_queue.get()
            if clip_feature is None:
                break
            
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
            result_queue.put(scores)

def main():
    args = parser.parse_args()
    args.anomaly_threshold = 0.5
    args.visual_width = 512
    args.p = 0.1
    args.visual_layers = 3
    args.attn_window = 16
    args.drop_rate = 0.5
    args.drop_attn_rate = 0.1
    args.drop_qkv_rate = 0.1
    args.cache_keep_max_len = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    WEIGHT_PATH = "./weights/xd_best.pth"
    VIDEO_SOURCE = 0
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
    result_queue = Queue(maxsize=5)

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model, feature_queue, result_queue, device, args)
    )
    inference_thread.start()
    print("[INFO] Inference thread started")

    try:
        video_stream = RealtimeVideoStream(
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

    current_score = 0.0
    print("="*50)
    print("StreamVAD Real-time Detection")
    print("Press 'q' to exit")
    print("="*50)

    while video_stream.running:
        clip_tensor = video_stream.read_clip()
        if clip_tensor is not None:
            clip_feature = clip_extractor.extract_clip_feature(clip_tensor)
            if not feature_queue.full():
                feature_queue.put(clip_feature)

        if not result_queue.empty():
            scores = result_queue.get()
            current_score = np.mean(scores)
            if current_score > args.anomaly_threshold:
                print(f"⚠️  Anomaly detected! Score: {current_score:.4f}")

        ret, frame = video_stream.cap.read()
        if ret:
            cv2.putText(frame, f"Score: {current_score:.4f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {args.anomaly_threshold:.4f}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Device: {device}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if current_score > args.anomaly_threshold:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)
                cv2.putText(frame, "ANOMALY DETECTED!", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("StreamVAD Real-time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    feature_queue.put(None)
    inference_thread.join()
    video_stream.stop()
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
    print("3. Press 'q' to exit")
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