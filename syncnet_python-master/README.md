## 相关依赖
```
pip install -r requirements.txt
In addition, `ffmpeg` is required.
```

## 相关文件介绍
'''
out_videos：可以存放由audio2head模型读取参考视频第一帧和参考视频音频运行产生的输出视频
videos：可以存放参考视频
'''

## 评价标准计算
```
计算PSNR：
python PSNR.py /path/to/your/ref_video /path/to/your/out_video

计算NIQE：
python NIQE.py /path/to/your/ref_video

计算SSIM：
python SSIM.py /path/to/your/ref_video /path/to/your/out_video

计算FID：
python FID.py /path/to/your/ref_video /path/to/your/out_video

计算LSE-C和LSE-D：
python run_pipeline.py --videofile /path/to/your/video --reference wav2lip --data_dir tmp_dir
python calculate_scores_real_videos.py --videofile /path/to/you/video --reference wav2lip --data_dir tmp_dir >> all_scores.txt
LSE-C和LSE-D会保存至all_scores.txt中，第一个是LSE-D，第二个是LSE-C
```