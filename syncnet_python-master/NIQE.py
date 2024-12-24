import cv2
import numpy as np
import cupy as cp
import scipy.io
from os.path import dirname, join
from PIL import Image
import scipy.special
import math
import os
from skimage.transform import resize
import sys
import cupyx.scipy.ndimage as cupyx_ndimage  # 导入cupyx的ndimage模块

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

def aggd_features(imdata):
    # 将imdata转移到GPU
    imdata = cp.asarray(imdata)
    # 展平imdata
    imdata = imdata.flatten()
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    
    left_mean_sqrt = cp.sqrt(cp.average(left_data)) if left_data.size > 0 else 0
    right_mean_sqrt = cp.sqrt(cp.average(right_data)) if right_data.size > 0 else 0

    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else cp.inf

    imdata2_mean = cp.mean(imdata2)
    r_hat = (cp.mean(cp.abs(imdata)) ** 2) / cp.mean(imdata2) if imdata2_mean != 0 else cp.inf
    rhat_norm = r_hat * (((gamma_hat ** 3) + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2) + 1) ** 2

    pos = cp.argmin((cp.asarray(prec_gammas) - rhat_norm) ** 2).get()
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = math.sqrt(gam1) / math.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt.get()
    br = aggdratio * right_mean_sqrt.get()

    N = (br - bl) * (gam2 / gam1)
    return (alpha, N, bl, br, left_mean_sqrt.get(), right_mean_sqrt.get())

def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = cp.var(imdata)
    E = cp.mean(cp.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = cp.argmin(cp.abs(nr_gam - rho)).get()
    return gamma_range[pos], sigma_sq.get()

def paired_product(new_im):
    # 将数据转移到CPU进行滚动操作
    new_im_cpu = new_im.get()
    shift1 = np.roll(new_im_cpu.copy(), 1, axis=1)
    shift2 = np.roll(new_im_cpu.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im_cpu.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im_cpu.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im_cpu
    V_img = shift2 * new_im_cpu
    D1_img = shift3 * new_im_cpu
    D2_img = shift4 * new_im_cpu

    return cp.asarray(H_img), cp.asarray(V_img), cp.asarray(D1_img), cp.asarray(D2_img)

def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum_weights = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum_weights += 2.0 * tmp
    weights = [w / sum_weights for w in weights]
    return cp.asarray(weights, dtype=cp.float32)

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(image.shape) == 2
    h, w = image.shape
    image = cp.asarray(image, dtype=cp.float32)
    
    # 使用cupyx.scipy.ndimage.correlate进行多维相关操作
    mu_image = cupyx_ndimage.correlate(image, avg_window[:, None], mode=extend_mode)
    mu_image = cupyx_ndimage.correlate(mu_image, avg_window[None, :], mode=extend_mode)
    
    var_image = cupyx_ndimage.correlate(image ** 2, avg_window[:, None], mode=extend_mode)
    var_image = cupyx_ndimage.correlate(var_image, avg_window[None, :], mode=extend_mode)
    var_image = cp.sqrt(cp.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image

def _niqe_extract_subband_feats(mscncoefs):
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return cp.asnumpy(cp.array([alpha_m, (bl + br) / 2.0,
                                 alpha1, N1, bl1, br1,  # (V)
                                 alpha2, N2, bl2, br2,  # (H)
                                 alpha3, N3, bl3, bl3,  # (D1)
                                 alpha4, N4, bl4, bl4,  # (D2)
                                 ]))

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = cp.asarray(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = cp.asarray(patch_features)
    return cp.asnumpy(patch_features)

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = img.shape
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # 确保补丁能够均匀划分
    hoffset = h % patch_size
    woffset = w % patch_size

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(cp.float32)
    # 使用skimage在CPU上调整图像大小
    img_cpu = cp.asnumpy(img)
    img2_cpu = resize(img_cpu, (int(img_cpu.shape[0] * 0.5), int(img_cpu.shape[1] * 0.5)), mode='constant', anti_aliasing=True)
    img2 = cp.asarray(img2_cpu, dtype=cp.float32)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(cp.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(cp.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))
    return feats

def niqe(inputImgData):
    patch_size = 96
    module_path = dirname(__file__)

    # 加载预训练的NIQE参数
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = cp.asarray(np.ravel(params["pop_mu"]))
    pop_cov = cp.asarray(params["pop_cov"])

    M, N = inputImgData.shape

    assert M > (patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    # 将图像转移到GPU
    inputImgData = cp.asarray(inputImgData, dtype=cp.float32)

    feats = cp.asarray(get_patches_test_features(inputImgData, patch_size))
    sample_mu = cp.mean(feats, axis=0)
    sample_cov = cp.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinvmat = cp.linalg.pinv(covmat)
    niqe_score = cp.sqrt(cp.dot(cp.dot(X, pinvmat), X)).get()

    return niqe_score

def evaluate_video_with_niqe(video_path, log_file_path="niqe_log.txt"):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    niqe_values = []
    NIQE_sum = 0
    frame_count = 0

    # 设置设备为GPU（如果可用）
    device = cp.cuda.Device()
    device.use()

    # 打开日志文件以追加模式
    try:
        log_file = open(log_file_path, "a")
    except Exception as e:
        print(f"Error: Could not open log file {log_file_path}: {e}")
        cap.release()
        return
    
    # 写入日志文件的头部（如果文件是空的）
    if os.path.getsize(log_file_path) == 0:
        log_file.write("Video_Path,Frame_Count,Average_NIQE\n")
    
    print("Processing frames and calculating NIQE...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将帧从BGR转换为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算NIQE
        try:
            niqe_value = niqe(gray_frame)
        except Exception as e:
            print(f"Error during NIQE calculation on frame {frame_count + 1}: {e}")
            continue  # 跳过该帧并继续处理
        
        NIQE_sum += niqe_value
        frame_count += 1

        # 可选：打印每帧的NIQE分数
        # print(f"Frame {frame_count}: NIQE score = {niqe_value}")
    
    cap.release()
    
    if frame_count > 0:
        NIQE_mean = NIQE_sum / frame_count
        print(f"Average NIQE over {frame_count} frames: {NIQE_mean}")
        
        # 记录到日志文件
        log_entry = f"\"{video_path}\",{frame_count},{NIQE_mean}\n"
        log_file.write(log_entry)
    else:
        print("No frames were processed.")
    
    log_file.close()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script_name.py <video_path> [<log_file_path>]")
        sys.exit(1)
    video_path = sys.argv[1]
    
    # 可选：获取日志文件路径
    if len(sys.argv) == 3:
        log_file_path = sys.argv[2]
    else:
        log_file_path = "niqe_log.txt"
    
    evaluate_video_with_niqe(video_path, log_file_path)
    print(f"NIQE scores logged to '{log_file_path}'.")

