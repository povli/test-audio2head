import cv2
import numpy as np
import cupy as cp
import sys
import os

def compute_psnr_gpu(ref_gray, out_gray):
    """
    在GPU上计算PSNR。
    """
    # 将图像数据转换为float32类型并传输到GPU
    ref_gpu = cp.asarray(ref_gray, dtype=cp.float32)
    out_gpu = cp.asarray(out_gray, dtype=cp.float32)
    
    # 计算均方误差 (MSE)
    mse = cp.mean((ref_gpu - out_gpu) ** 2)
    
    if mse == 0:
        return float('inf')
    
    PIXEL_MAX = 255.0
    psnr_value = 20 * cp.log10(PIXEL_MAX / cp.sqrt(mse))
    
    # 将结果从GPU传回CPU
    return psnr_value.get()

def evaluate_video_with_psnr(reference_video_path, output_video_path, log_file_path="psnr_log.txt"):
    # 打开参考视频和输出视频文件
    ref_cap = cv2.VideoCapture(reference_video_path)
    out_cap = cv2.VideoCapture(output_video_path)
    
    if not ref_cap.isOpened() or not out_cap.isOpened():
        print(f"Error: Could not open one of the video files.")
        return
    
    # 获取视频属性
    ref_frame_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_frame_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化变量
    frame_count = 0
    psnr_scores = []
    
    # 打开日志文件以追加模式
    try:
        log_file = open(log_file_path, "a")
    except Exception as e:
        print(f"Error: Could not open log file {log_file_path}: {e}")
        ref_cap.release()
        out_cap.release()
        return
    
    # 写入日志文件的头部（如果文件是空的）
    if os.path.getsize(log_file_path) == 0:
        log_file.write("Reference_Video_Path,Output_Video_Path,Frame_Count,Average_PSNR\n")
    
    while True:
        ret_ref, ref_frame = ref_cap.read()
        ret_out, out_frame = out_cap.read()
        
        if not ret_ref:
            # 参考视频已经读完，但输出视频可能还有剩余帧（不处理这些帧）
            break
        
        if not ret_out:
            break
        
        # 调整输出帧大小以匹配参考帧
        out_frame_resized = cv2.resize(out_frame, (ref_frame_width, ref_frame_height))
        
        # 将BGR图像转换为灰度图像
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        out_gray = cv2.cvtColor(out_frame_resized, cv2.COLOR_BGR2GRAY)
        
        # 使用GPU计算PSNR
        score = compute_psnr_gpu(ref_gray, out_gray)
        
        # 保存PSNR分数
        psnr_scores.append(score)
        
        frame_count += 1
        
        # 可选：打印每帧的PSNR分数
        # print(f"Frame {frame_count}: PSNR score = {score}")
    
    # 计算并打印平均PSNR分数
    if psnr_scores:
        average_score = np.mean(psnr_scores)
        output_message = f"Average PSNR score over {frame_count} frames: {average_score}"
        print(output_message)
        
        # 写入日志文件
        log_entry = f"\"{reference_video_path}\",\"{output_video_path}\",Average PSNR score over{frame_count},frames:{average_score}\n"
        log_file.write(log_entry)
    else:
        print("No frames were processed.")
    
    # 关闭日志文件
    log_file.close()
    
    # 释放视频捕获对象
    ref_cap.release()
    out_cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python script_name.py <reference_video_path> <output_video_path> [<log_file_path>]")
        sys.exit(1)
    reference_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    
    # 可选：获取日志文件路径
    if len(sys.argv) == 4:
        log_file_path = sys.argv[3]
    else:
        log_file_path = "psnr_log.txt"
    
    evaluate_video_with_psnr(reference_video_path, output_video_path, log_file_path)
