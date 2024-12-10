import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
import lpips
from sewar.full_ref import uqi
import torch
import torchvision.transforms as transforms

# 加载图像
def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)  # 确保以 RGB 格式读取
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 计算 PSNR
def calculate_psnr(imageA, imageB):
    mse = normalized_root_mse(imageA, imageB) ** 2
    if mse == 0:
        return float('inf')  # 如果 MSE 为零，返回无限大的 PSNR
    psnr = peak_signal_noise_ratio(imageA, imageB, data_range=imageA.max() - imageA.min())
    return psnr

# 计算 SSIM
def calculate_ssim(imageA, imageB):
    win_size = 7  # 固定 win_size，适用于大多数情况
    ssim, _ = structural_similarity(imageA, imageB, full=True, channel_axis=-1, win_size=win_size)
    return ssim

# 计算 LPIPS
def calculate_lpips(imageA, imageB, lpips_model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    imageA = transform(imageA).unsqueeze(0).cuda()
    imageB = transform(imageB).unsqueeze(0).cuda()
    lpips_value = lpips_model(imageA, imageB)
    return lpips_value.item()

# 计算 NRMSE
def calculate_nrmse(imageA, imageB):
    nrmse_value = normalized_root_mse(imageA, imageB)
    return nrmse_value

# 计算 UQI
def calculate_uqi(imageA, imageB):
    uqi_value = uqi(imageA, imageB)
    return uqi_value

# 文件夹路径
reference_folder = 'target_real'
generated_folder = 'target_fake'

# 获取文件列表
reference_images = sorted(os.listdir(reference_folder))
generated_images = sorted(os.listdir(generated_folder))

# 初始化 LPIPS 模型
lpips_model = lpips.LPIPS(net='alex').cuda()

# 初始化列表以存储每个图像对的指标
psnr_list = []
ssim_list = []
lpips_list = []
nrmse_list = []
uqi_list = []

# 初始化日志列表以存储配对信息
pairing_log = []

# 逐对图像计算指标
for img_name in reference_images:
    ref_img_path = os.path.join(reference_folder, img_name)
    gen_img_path = os.path.join(generated_folder, img_name)

    if os.path.exists(gen_img_path):
        # 记录配对信息
        pairing_log.append(f"Reference image: {img_name}, Generated image: {img_name}")

        # 加载图像
        ref_img = load_image(ref_img_path)
        gen_img = load_image(gen_img_path)

        # 计算指标
        psnr = calculate_psnr(ref_img, gen_img)
        ssim = calculate_ssim(ref_img, gen_img)
        lpips_value = calculate_lpips(ref_img, gen_img, lpips_model)
        nrmse_value = calculate_nrmse(ref_img, gen_img)
        uqi_value = calculate_uqi(ref_img, gen_img)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_value)
        nrmse_list.append(nrmse_value)
        uqi_list.append(uqi_value)
    else:
        print(f"Generated image for {img_name} not found.")

# 打印配对信息
for log in pairing_log:
    print(log)

# 计算平均值
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)
mean_lpips = np.mean(lpips_list)
mean_nrmse = np.mean(nrmse_list)
mean_uqi = np.mean(uqi_list)

print(f'Average PSNR: {mean_psnr}')
print(f'Average SSIM: {mean_ssim}')
print(f'Average LPIPS: {mean_lpips}')
print(f'Average NRMSE: {mean_nrmse}')
print(f'Average UQI: {mean_uqi}')
