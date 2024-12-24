import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import sys

def calculate_video_ssim(ref_video_path, out_video_path):
    # 打开视频文件
    ref_cap = cv2.VideoCapture(ref_video_path)
    out_cap = cv2.VideoCapture(out_video_path)

    # 获取视频帧的宽度和高度（假设两者相同）
    ref_frame_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_frame_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化SSIM列表
    ssim_scores = []

    while True:
        # 读取参考视频和输出视频的帧
        ret_ref, ref_frame = ref_cap.read()
        ret_out, out_frame = out_cap.read()

        # 如果输出视频已经结束，则停止循环（即使参考视频还有帧）
        if not ret_out:
            break

        # 如果参考视频已经结束（理论上不应该发生，因为参考视频应该更长），则也停止循环
        if not ret_ref:
            print("Warning: Reference video ended before output video, which is unexpected.")
            break

        # 转换帧的颜色空间（如果需要，这里假设已经是RGB或灰度图，OpenCV默认读取为BGR）
        # ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)  # 如果需要RGB
        # out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)  # 如果需要RGB
        out_frame_resized = cv2.resize(out_frame, (ref_frame_width, ref_frame_height))
        # 由于OpenCV读取的是BGR，而skimage.metrics.structural_similarity期望的是RGB或灰度图，
        # 如果你的SSIM计算库需要Rcce7aedf909d531d03df263977188917B，请取消上面两行的注释，并注释掉下面的转换（这里我们假设使用灰度图）
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        out_gray = cv2.cvtColor(out_frame_resized, cv2.COLOR_BGR2GRAY)

        # 计算SSIM
        score, _ = ssim(ref_gray, out_gray, full=True)
        ssim_scores.append(score)

    # 处理输出视频比参考视频少一帧的情况（实际上在这个循环中不需要额外处理，因为循环会在输出视频结束时停止）

    # 计算平均SSIM
    average_ssim = np.mean(ssim_scores)

    # 释放视频捕获对象
    ref_cap.release()
    out_cap.release()

    return average_ssim, ssim_scores

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <reference_video_path> <output_video_path>")
        sys.exit(1)
    ref_video_path = sys.argv[1]
    out_video_path = sys.argv[2]
    average_ssim, ssim_scores = calculate_video_ssim(ref_video_path, out_video_path)
    print(f"Average SSIM: {average_ssim}")
    # 如果需要查看每一帧的SSIM，可以打印ssim_scores列表
    # for i, score in enumerate(ssim_scores):
    #     print(f"Frame {i+1} SSIM: {score}")