#!/bin/bash

# 定义文件夹路径
IMG_DIR="/app/Audio2Head/Audio2Head/input_img"
WAV_DIR="/app/Audio2Head/Audio2Head/input_wav"
OUT_DIR="/app/out_video"

# 确保工作目录正确
cd /app/Audio2Head/Audio2Head || exit
# 创建输出目录（如果不存在）
mkdir -p "$OUT_DIR"

# 支持的图片格式
IMG_EXTENSIONS=("png" "jpg" "jpeg")

# 遍历所有音频文件
for audio_path in "$WAV_DIR"/*.wav; do
    # 检查是否存在.wav文件
    if [ ! -e "$audio_path" ]; then
        echo "No .wav files found in $WAV_DIR"
        exit 1
    fi

    # 提取音频文件的基名（不含扩展名）
    audio_filename=$(basename "$audio_path")
    base_name="${audio_filename%.*}"

    # 初始化图片文件变量
    img_file=""
    
    # 查找匹配的图片文件
    for ext in "${IMG_EXTENSIONS[@]}"; do
        potential_img="$IMG_DIR/$base_name.$ext"
        if [ -f "$potential_img" ]; then
            img_file="$potential_img"
            img_extension="$ext"
            break
        fi
    done

    # 如果未找到匹配的图片文件，输出警告并跳过
    if [ -z "$img_file" ]; then
        echo "Warning: No matching image file found for $audio_filename in $IMG_DIR"
        continue
    fi

    # 提取图片文件名
    img_filename=$(basename "$img_file")

    # 输出当前处理的文件
    echo "Processing: $audio_filename and $img_filename"

    # 运行Python脚本
    # 修改以下部分
    python /app/Audio2Head/Audio2Head/inference.py --audio_filename "$audio_filename" --img_filename "$img_filename"


    # 检查Python脚本是否成功执行
    if [ $? -ne 0 ]; then
        echo "Error: Processing failed for $audio_filename and $img_filename"
        continue
    fi

    echo "Successfully processed $audio_filename and $img_filename"
done

echo "All matching files have been processed."
