import torch
from torch.utils.data import DataLoader
from model import StreamVADWithXL
from dataset import UCFDataset, XDDataset
import argparse
from sklearn.metrics import roc_auc_score
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="StreamVAD+XL Testing")
    parser.add_argument('--dataset', type=str, default='ucf', choices=['ucf', 'xd'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mem_len', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def test(args):
    print(f"测试数据集: {args.dataset}")
    print(f"加载模型: {args.model_path}")
    
    # 加载模型
    model = StreamVADWithXL(mem_len=args.mem_len).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    
    # 加载测试集
    if args.dataset == 'ucf':
        test_dataset = UCFDataset(args.test_list, args.data_root, is_train=False)
    else:
        test_dataset = XDDataset(args.test_list, args.data_root, is_train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (video_features, _, frame_labels) in enumerate(test_loader):
            video_features = video_features.to(args.device)
            video_features = torch.nn.functional.normalize(video_features, p=2, dim=-1)
            
            B, Total_T, D = video_features.shape
            memories = None
            video_scores = []
            
            for t in range(0, Total_T, args.chunk_size):
                chunk = video_features[:, t:t+args.chunk_size, :]
                if chunk.size(1) == 0:
                    continue
                
                scores, memories, _ = model(chunk, memories)
                video_scores.append(scores.cpu())
            
            video_scores = torch.cat(video_scores, dim=1).numpy()
            frame_labels = frame_labels.numpy()
            
            all_scores.append(video_scores.flatten())
            all_labels.append(frame_labels.flatten())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx+1}/{len(test_loader)} 个视频")
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    auc = roc_auc_score(all_labels, all_scores)
    print(f"\n测试完成！AUC: {auc:.4f}")

if __name__ == "__main__":
    args = parse_args()
    test(args)