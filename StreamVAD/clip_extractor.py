import clip
import torch

class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_clip_feature(self, clip_tensor):
        batch_size, clip_len, c, h, w = clip_tensor.shape
        frames = clip_tensor.view(-1, c, h, w).to(self.device)
        features = self.model.encode_image(frames)
        features = features.view(batch_size, clip_len, -1)
        return features.cpu().numpy()
