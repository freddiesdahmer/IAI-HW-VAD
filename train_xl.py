import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from model import StreamVADWithXL
# 导入官方原版数据集类
from dataset import UCFDataset_Whole_test_train, XDDataset_Whole_test_train
import argparse
import os

os.makedirs("checkpoints", exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="StreamVAD+Transformer-XL Training")
    
    parser.add_argument('--dataset', type=str, default='ucf', choices=['ucf', 'xd'], help='训练数据集')
    parser.add_argument('--train_list', type=str, required=True, help='训练集csv路径')
    
    parser.add_argument('--mem_len', type=int, default=32, help='记忆缓存最大长度')
    parser.add_argument('--chunk_size', type=int, default=16, help='流式处理片段长度')
    parser.add_argument('--attnwindow', type=int, default=256, help='官方默认窗口大小')
    
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()

def mil_loss(scores, labels):
    topk_scores, _ = torch.topk(scores, k=min(10, scores.size(1)), dim=1)
    video_pred = torch.mean(topk_scores, dim=1)
    bce_loss = torch.nn.functional.binary_cross_entropy(video_pred, labels.float())
    smooth_loss = torch.mean(torch.abs(scores[:, 1:] - scores[:, :-1])) if scores.size(1) > 1 else 0.0
    return bce_loss + 0.1 * smooth_loss

def train(args):
    print(f"使用设备: {args.device}")
    print(f"训练数据集: {args.dataset}")

    # 初始化模型
    model = StreamVADWithXL(mem_len=args.mem_len).to(args.device)
    
    # 权重初始化
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    model.apply(weights_init)

    # ===================== 加载官方原版数据集 =====================
    if args.dataset == 'ucf':
        # 加载正常视频 + 异常视频（官方分开加载）
        normal_dataset = UCFDataset_Whole_test_train(
            file_path=args.train_list, test_mode=False, normal=True, attnwindow=args.attnwindow
        )
        abnormal_dataset = UCFDataset_Whole_test_train(
            file_path=args.train_list, test_mode=False, normal=False, attnwindow=args.attnwindow
        )
        train_dataset = ConcatDataset([normal_dataset, abnormal_dataset])
    else:
        train_dataset = XDDataset_Whole_test_train(
            file_path=args.train_list, test_mode=False, attnwindow=args.attnwindow
        )

    print(f"数据集总数量: {len(train_dataset)}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            # 官方数据集输出：(feature, label_str, length, name)
            clip_feature, label_str, _, _ = data
            
            # 修复dtype：将可能为float16的特征转为float32，与模型权重一致
            clip_feature = clip_feature.float().to(args.device)
            
            # 标签转换：Normal=0，其他=1（标签为字符串，需转为0/1张量）
            labels = torch.tensor([0 if l == 'Normal' else 1 for l in label_str]).to(args.device)
            
            # 特征归一化（对最后一维做L2归一化）
            clip_feature = torch.nn.functional.normalize(clip_feature, p=2, dim=-1)

            # 流式前向传播
            B, Total_T, D = clip_feature.shape
            memories = None
            batch_loss = 0.0
            
            for t in range(0, Total_T, args.chunk_size):
                chunk = clip_feature[:, t:t+args.chunk_size, :]
                scores, memories, _ = model(chunk, memories)
                
                loss = mil_loss(scores, labels)
                loss.backward()
                batch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_loss += batch_loss

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {batch_loss:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} 平均损失: {avg_loss:.4f}\n")
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.dataset}_xl_epoch{epoch+1}.pth")

if __name__ == "__main__":
    args = parse_args()
    train(args)