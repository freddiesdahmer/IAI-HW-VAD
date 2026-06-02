import clip
import torch
import torch.nn.functional as F


class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/16", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_clip_feature(self, pil_images):
        """
        接收原始 PIL Image 列表，内部用 CLIP 官方 preprocess 处理
        Args:
            pil_images: List[PIL.Image]，长度为 clip_length（如16）
        Returns:
            features: 形状 (1, T, 512) 的 tensor
        """
        features = []
        for pil_img in pil_images:
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(image_input)  # (1, 512)
            feat = F.normalize(feat, p=2, dim=-1)        # L2 归一化
            features.append(feat.squeeze(0))              # (512,)

        features = torch.stack(features, dim=0).unsqueeze(0)  # (1, T, 512)
        return features
