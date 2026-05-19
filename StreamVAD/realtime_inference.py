import torch
import cv2
import numpy as np
import threading
from queue import Queue
from model import LSHMA as StreamVAD
from xd_option import parser  # 使用UCF权重请改为 ucf_option
from realtime_input import RealtimeVideoStream
from clip_extractor import CLIPFeatureExtractor

def inference_worker(model, feature_queue, result_queue, device, args):
    model.eval()
    # 实时推理使用固定视频标识（单流模式）
    video_name = torch.tensor([0], device=device)  # 用数字代替字符串，避免设备不匹配
    
    with torch.no_grad():
        while True:
            clip_feature = feature_queue.get()
            if clip_feature is None:
                break
            
            # 转换为模型要求的输入格式 [batch, clip_len, feature_dim]
            input_feature = torch.from_numpy(clip_feature).to(device)
            batch_size, clip_len, _ = input_feature.shape
            
            # 调用LSHMA的Test模式（内部自动管理缓存）
            logits = model(
                visual=input_feature,
                flag='Test',
                lengths=clip_len,
                video_names=video_name,
                threshold=args.anomaly_threshold  # 传入阈值控制缓存清理
            )
            
            # 转换为异常分数（sigmoid已在模型内部完成）
            scores = logits.squeeze().cpu().numpy()
            result_queue.put(scores)

def main():
    args = parser.parse_args()
    # 添加实时推理所需的额外参数
    args.anomaly_threshold = 0.5  # XD-Violence推荐0.5，UCF-Crime推荐0.4
    args.visual_width = 512       # CLIP特征维度
    args.p = 0.1                  # 自注意力权重系数（官方默认值）
    args.visual_layers = 3        # Transformer层数（官方默认值）
    args.attn_window = 16         # 注意力窗口大小（与clip_length一致）
    args.drop_rate = 0.5
    args.drop_attn_rate = 0.1
    args.drop_qkv_rate = 0.1
    args.cache_keep_max_len = 64  # 最大缓存长度（官方默认值）
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ====================== 请修改以下配置 ======================
    WEIGHT_PATH = "./weights/xd_best.pth"  # 预训练权重路径
    VIDEO_SOURCE = 0  # 0=本地摄像头，RTSP地址如 "rtsp://admin:123456@192.168.1.100:554"
    FRAME_INTERVAL = 1  # 帧采样间隔（增大可降低计算量）
    # ==========================================================

    # 初始化模型（使用官方参数）
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

    # 加载权重（添加strict=False解决不匹配问题）
    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("[SUCCESS] 模型加载成功")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        print("[提示] 请检查权重路径是否正确，已下载预训练权重并放在weights文件夹")
        sys.exit(1)

    # 初始化队列
    feature_queue = Queue(maxsize=5)
    result_queue = Queue(maxsize=5)

    # 启动推理线程
    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model, feature_queue, result_queue, device, args)
    )
    inference_thread.start()
    print("[INFO] 推理线程已启动")

    # 初始化视频流和特征提取器
    try:
        video_stream = RealtimeVideoStream(
            source=VIDEO_SOURCE, 
            clip_length=args.attn_window, 
            frame_interval=FRAME_INTERVAL
        ).start()
        clip_extractor = CLIPFeatureExtractor(device=device)
        print("[SUCCESS] 视频流和特征提取器初始化成功")
    except Exception as e:
        print(f"[ERROR] 视频流初始化失败: {e}")
        print("[提示] 请检查摄像头权限或RTSP地址是否正确")
        sys.exit(1)

    current_score = 0.0
    print("="*50)
    print("StreamVAD 实时推理已启动")
    print("按 'q' 键退出程序")
    print("="*50)

    while video_stream.running:
        # 主线程：视频捕获和特征提取
        clip_tensor = video_stream.read_clip()
        if clip_tensor is not None:
            clip_feature = clip_extractor.extract_clip_feature(clip_tensor)
            if not feature_queue.full():
                feature_queue.put(clip_feature)

        # 非阻塞获取推理结果
        if not result_queue.empty():
            scores = result_queue.get()
            current_score = np.mean(scores)
            if current_score > args.anomaly_threshold:
                print(f"⚠️  检测到异常！异常分数: {current_score:.4f}")

        # 实时可视化
        ret, frame = video_stream.cap.read()
        if ret:
            # 绘制信息
            cv2.putText(frame, f"Score: {current_score:.4f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {args.anomaly_threshold:.4f}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 异常报警
            if current_score > args.anomaly_threshold:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)
                cv2.putText(frame, "ANOMALY DETECTED!", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("StreamVAD Real-time Detection", frame)

        # 退出控制
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理资源
    feature_queue.put(None)
    inference_thread.join()
    video_stream.stop()
    print("[INFO] 程序已正常退出")

if __name__ == "__main__":
    main()