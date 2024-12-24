import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import sqrtm
from torchvision import models, transforms
from PIL import Image
import sys
import os
from datetime import datetime

# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的InceptionV3模型，并去掉最后的分类层
inception_model = models.inception_v3(pretrained=True).to(device)
inception_model.fc = torch.nn.Identity()  # 去掉最后的分类层
inception_model.eval()

# 定义图像预处理函数
preprocess = transforms.Compose([
    transforms.Resize(299),  # InceptionV3的输入大小
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 计算图像特征
def get_features_batch(images, batch_size=32):
    features = []
    num_images = len(images)
    for i in range(0, num_images, batch_size):
        batch_images = images[i:i+batch_size]
        batch_pil = [Image.fromarray(img) for img in batch_images]
        batch_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(device)
        with torch.no_grad():
            batch_features = inception_model(batch_tensor)
        features.append(batch_features.cpu().numpy())
    return np.vstack(features)

# 计算FID分数
def calculate_fid(features1, features2):
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 日志记录函数
def log_fid(log_file_path, ref_video_path, out_video_path, frame_count, fid_score):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp},\"{ref_video_path}\",\"{out_video_path}\",{frame_count},{fid_score}\n"
    with open(log_file_path, "a") as log_file:
        # 如果文件是空的，写入表头
        if os.path.getsize(log_file_path) == 0:
            log_file.write("Timestamp,Reference_Video_Path,Output_Video_Path,Frame_Count,FID_Score\n")
        log_file.write(log_entry)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python script_name.py <reference_video_path> <output_video_path> [<log_file_path>]")
        sys.exit(1)
    
    ref_video_path = sys.argv[1]
    out_video_path = sys.argv[2]
    
    # 可选：获取日志文件路径
    if len(sys.argv) == 4:
        log_file_path = sys.argv[3]
    else:
        log_file_path = "fid_log.txt"
    
    # 检查视频文件是否存在
    if not os.path.isfile(ref_video_path):
        print(f"Error: Reference video file '{ref_video_path}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(out_video_path):
        print(f"Error: Output video file '{out_video_path}' does not exist.")
        sys.exit(1)
    
    # 打开参考视频和输出视频文件
    ref_cap = cv2.VideoCapture(ref_video_path)
    out_cap = cv2.VideoCapture(out_video_path)

    if not ref_cap.isOpened():
        print(f"Error: Could not open reference video '{ref_video_path}'.")
        sys.exit(1)
    if not out_cap.isOpened():
        print(f"Error: Could not open output video '{out_video_path}'.")
        ref_cap.release()
        sys.exit(1)
    
    # 获取视频属性
    ref_frame_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_frame_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ref_frames = []
    out_frames = []
    
    print("Reading frames from videos...")
    
    while True:
        ret_ref, ref_frame = ref_cap.read()
        ret_out, out_frame = out_cap.read()
        
        if not ret_ref or not ret_out:
            break
        # 转换为RGB，因为Inception模型使用RGB图像
        ref_frames.append(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
        out_frames.append(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
    
    # 处理输出视频比参考视频少一帧的情况
    if len(out_frames) < len(ref_frames):
        ref_frames = ref_frames[:len(out_frames)]
    elif len(ref_frames) < len(out_frames):
        out_frames = out_frames[:len(ref_frames)]
    
    frame_count = len(ref_frames)
    print(f"Total frames to process: {frame_count}")
    
    # 释放视频捕获对象
    ref_cap.release()
    out_cap.release()
    
    if frame_count == 0:
        print("No frames to process.")
        sys.exit(1)
    
    print("Extracting features from reference video frames...")
    ref_features = get_features_batch(ref_frames)
    
    print("Extracting features from output video frames...")
    out_features = get_features_batch(out_frames)
    
    print("Calculating FID score...")
    fid_score = calculate_fid(ref_features, out_features)
    print(f"FID score: {fid_score}")
    
    # 记录到日志文件
    log_fid(log_file_path, ref_video_path, out_video_path, frame_count, fid_score)
    print(f"FID score logged to '{log_file_path}'.")
