import os
import subprocess
import logging
from multiprocessing import Pool, cpu_count

def get_ref_video_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.mp4')]

def setup_logging(log_file='batch_process.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_command(command, logger):
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"命令成功执行: {' '.join(command)}")
        logger.info(f"输出:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"错误输出:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {' '.join(command)}")
        logger.error(f"错误输出:\n{e.stderr}")

def process_video_pair(args):
    ref_path, out_path, logger = args
    ref_filename = os.path.basename(ref_path)
    out_filename = os.path.basename(out_path)

    logger.info(f"正在处理: {ref_filename} 与 {out_filename}")

    # 1. 运行 PSNR.py
    psnr_command = ['python', 'PSNR.py', ref_path, out_path]
    run_command(psnr_command, logger)

    # 2. 运行 NIQE.py
    niqe_command = ['python', 'NIQE.py', out_path]
    run_command(niqe_command, logger)

    # 3. 运行 SSIM.py
    ssim_command = ['python', 'SSIM.py', ref_path, out_path]
    run_command(ssim_command, logger)

    # 4. 运行 FID.py
    fid_command = ['python', 'FID.py', ref_path, out_path]
    run_command(fid_command, logger)

    # 5. 运行 run_pipeline.py
    run_pipeline_command = [
        'python', 'run_pipeline.py',
        '--videofile', out_path,
        '--reference', 'wav2lip',
        '--data_dir', 'tmp_dir'
    ]
    run_command(run_pipeline_command, logger)

    # 6. 运行 calculate_scores_real_videos.py 并将输出追加到 all_scores.txt
    calculate_scores_command = [
        'python', 'calculate_scores_real_videos.py',
        '--videofile', out_path,
        '--reference', 'wav2lip',
        '--data_dir', 'tmp_dir'
    ]
    try:
        result = subprocess.run(
            calculate_scores_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"命令成功执行: {' '.join(calculate_scores_command)}")
        logger.info(f"输出:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"错误输出:\n{result.stderr}")
        
        # 追加输出到 all_scores.txt
        all_scores_path = os.path.join(os.getcwd(), 'all_scores.txt')
        with open(all_scores_path, 'a') as f:
            f.write(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {' '.join(calculate_scores_command)}")
        logger.error(f"错误输出:\n{e.stderr}")

def main(ref_dir, out_dir):
    setup_logging()
    logger = logging.getLogger()

    if not os.path.isdir(ref_dir):
        logger.error(f"参考视频目录不存在: {ref_dir}")
        return
    if not os.path.isdir(out_dir):
        logger.error(f"输出视频目录不存在: {out_dir}")
        return

    ref_files = get_ref_video_files(ref_dir)
    
    if not ref_files:
        logger.error("参考视频目录下没有找到任何.mp4文件。")
        return
    
    logger.info(f"找到 {len(ref_files)} 个参考文件。开始处理...")

    # 清空 all_scores.txt
    all_scores_path = os.path.join(os.getcwd(), 'all_scores.txt')
    with open(all_scores_path, 'w') as f:
        f.write('')  # 清空文件

    # 准备任务列表
    tasks = []
    for ref_filename in sorted(ref_files):
        ref_basename = os.path.splitext(ref_filename)[0]
        out_filename = f"{ref_basename}_{ref_basename}.mp4"
        ref_path = os.path.join(ref_dir, ref_filename)
        out_path = os.path.join(out_dir, out_filename)
        
        if not os.path.isfile(out_path):
            logger.warning(f"输出视频文件不存在: {out_filename}. 跳过.")
            continue
        
        tasks.append((ref_path, out_path, logger))

    # 使用进程池并行处理
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_video_pair, tasks)

    logger.info("所有文件处理完成。")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_video_dir = os.path.join(script_dir, 'ref_video')
    out_video_dir = os.path.join(script_dir, 'out_video')
    
    main(ref_video_dir, out_video_dir)
