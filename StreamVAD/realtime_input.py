import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# 与VadCLIP完全一致的预处理（不可修改！）
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
        # 设置视频流参数（提升稳定性）
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
