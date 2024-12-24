import argparse
import subprocess
import python_speech_features
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import pyworld
import torch
from modules.audio2pose import get_pose_from_audio
from skimage import io, img_as_float32
import cv2
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.audio2kp import AudioModel3D
import yaml, os, imageio

def draw_annotation_box(image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=float).reshape(-1, 3)  # 如果需要使用 NumPy 的 float64 类型，可以改成 np.float64

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

def inter_pitch(y, y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while(i < frame_num):
        if y_flag[i] == 0:
            while True:
                if y_flag[i] == 0:
                    if i == frame_num-1:
                        if last != -1:
                            y[last+1:] = y[last]
                        i += 1
                        break
                    i += 1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i - last + 1
                fy = np.array([y[last], y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx, fy)
                fx_new = np.linspace(0, 1, inter_num)
                fy_new = f(fx_new)
                y[last+1:i] = fy_new[1:-1]
                last = i
                i += 1
        else:
            last = i
            i += 1
    return y

def get_audio_feature_from_audio(audio_path, norm=True):
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]
    if norm:
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        a = python_speech_features.mfcc(audio, sample_rate)
        b = python_speech_features.logfbank(audio, sample_rate)
        c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
        c_flag = (c == 0.0) ^ 1
        c = inter_pitch(c, c_flag)
        c = np.expand_dims(c, axis=1)
        c_flag = np.expand_dims(c_flag, axis=1)
        frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

        cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)
        return cat

def audio2head(audio_path, img_path, model_path, save_path):
    temp_audio = "./results/temp.wav"
    # 使用 ffmpeg 将输入音频转换为指定格式
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, temp_audio))
    output = subprocess.call(command, shell=True, stdout=None)

    # 读取转换后的音频特征
    audio_feature = get_audio_feature_from_audio(temp_audio)
    frames = len(audio_feature) // 4

    # 读取并处理图片
    img = io.imread(img_path)[:, :, :3]  # 读取图片，保留前三个通道（RGB）
    img = cv2.resize(img, (256, 256))   # 调整图片大小

    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()

    # 获取参考姿态
    ref_pose_rot, ref_pose_trans = get_pose_from_audio(img, audio_feature, model_path)
    torch.cuda.empty_cache()

    # 加载配置文件
    config_file = r"./config/vox-256.yaml"
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    # 加载参数
    opt = argparse.Namespace(**yaml.load(open("./config/parameters.yaml"), Loader=yaml.FullLoader))
    audio2kp = AudioModel3D(opt).cuda()

    # 加载预训练模型
    checkpoint = torch.load(model_path)
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    generator.load_state_dict(checkpoint["generator"])
    audio2kp.load_state_dict(checkpoint["audio2kp"])

    generator.eval()
    kp_detector.eval()
    audio2kp.eval()

    # 准备音频和姿态数据
    audio_f = []
    poses = []
    pad = np.zeros((4, 41), dtype=np.float32)
    for i in range(0, frames, opt.seq_len // 2):
        temp_audio = []
        temp_pos = []
        for j in range(opt.seq_len):
            if i + j < frames:
                temp_audio.append(audio_feature[(i+j)*4:(i+j)*4+4])
                trans = ref_pose_trans[i + j]
                rot = ref_pose_rot[i + j]
            else:
                temp_audio.append(pad)
                trans = ref_pose_trans[-1]
                rot = ref_pose_rot[-1]

            pose = np.zeros([256, 256])
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            temp_pos.append(pose)
        audio_f.append(temp_audio)
        poses.append(temp_pos)

    audio_f = torch.from_numpy(np.array(audio_f, dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0)

    bs = audio_f.shape[1]
    predictions_gen = []
    total_frames = 0

    for bs_idx in range(bs):
        t = {}

        t["audio"] = audio_f[:, bs_idx].cuda()
        t["pose"] = poses[:, bs_idx].cuda()
        t["id_img"] = img
        kp_gen_source = kp_detector(img)

        gen_kp = audio2kp(t)
        if bs_idx == 0:
            startid = 0
            end_id = opt.seq_len // 4 * 3
        else:
            startid = opt.seq_len // 4
            end_id = opt.seq_len // 4 * 3

        for frame_bs_idx in range(startid, end_id):
            tt = {}
            tt["value"] = gen_kp["value"][:, frame_bs_idx]
            if opt.estimate_jacobian:
                tt["jacobian"] = gen_kp["jacobian"][:, frame_bs_idx]
            out_gen = generator(img, kp_source=kp_gen_source, kp_driving=tt)
            out_gen["kp_source"] = kp_gen_source
            out_gen["kp_driving"] = tt
            del out_gen['sparse_deformed']
            del out_gen['occlusion_map']
            del out_gen['deformed']
            predictions_gen.append(
                (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))

            total_frames += 1
            if total_frames >= frames:
                break
        if total_frames >= frames:
            break

    log_dir = save_path
    temp_dir = os.path.join(log_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    image_name = os.path.basename(img_path)[:-4] + "_" + os.path.basename(audio_path)[:-4] + ".mp4"

    video_path = os.path.join(temp_dir, image_name)

    # 生成视频文件
    imageio.mimsave(video_path, predictions_gen, format='FFMPEG', fps=25.0)

    # 将音频合并到视频中
    save_video = os.path.join(log_dir, image_name)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
    os.system(cmd)
    os.remove(video_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将音频和图片转换为视频")

    # 修改音频和图片路径，固定在input_wav和input_img文件夹下
    parser.add_argument("--audio_filename", default="intro.wav", help="音频文件名，位于 ./input_wav/ 目录下")
    parser.add_argument("--img_filename", default="paint.jpg", help="图片文件名，位于 ./input_img/ 目录下")
    # 修改 save_path 的默认值为上两级目录的 out_video 文件夹
    parser.add_argument("--save_path", default=os.path.join("..", "..", "out_video"), help="保存路径")
    parser.add_argument("--model_path", default=r"/app/Audio2Head/Audio2Head/checkpoints/audio2head.pth.tar", help="预训练模型路径")

    parse = parser.parse_args()

    # 构建完整的音频和图片路径
    audio_path = os.path.join("/app/Audio2Head/Audio2Head/input_wav", parse.audio_filename)
    img_path = os.path.join("/app/Audio2Head/Audio2Head/input_img", parse.img_filename)


    # 检查文件是否存在
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音频文件未找到: {audio_path}")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"图片文件未找到: {img_path}")

    os.makedirs(parse.save_path, exist_ok=True)
    audio2head(audio_path, img_path, parse.model_path, parse.save_path)
