import numpy as np
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import os
import lpips
import torchvision.transforms as transforms
from os import listdir

# 添加项目路径到sys.path以解决导入问题
sys.path.append(r"D:\cvpr25_3D-GSW")
from utils.general_utils import safe_state
from decoder.util.ssim import ssim

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 图像转换
img_transform = transforms.ToTensor()

# 初始化LPIPS网络 - 根据设备内存情况选择在CPU或GPU上运行
lpips_device = 'cpu' if device.type == 'cuda' else device  # 如果是GPU但内存不足，使用CPU
lpips_net = lpips.LPIPS(net='vgg').to(lpips_device)

# 路径设置
benchmark_dir = r"D:\latest\new way\output\replace_cαsr\evaluate\train"
test_dir = r"D:\latest\new way\output\replace_cαsr\evaluate\render"
output_dir = r"D:\latest\new way\output\replace_cαsr\evaluate"

# 获取图片列表
benchmark_files = sorted([f for f in listdir(benchmark_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
test_files = sorted([f for f in listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

# 确保图片数量一致
assert len(benchmark_files) == len(test_files), f"基准图片数量 ({len(benchmark_files)}) 与测试图片数量 ({len(test_files)}) 不一致"

# 初始化统计变量
total_ssim = 0.0
total_psnr = 0.0
total_lpips = 0.0
num_images = len(benchmark_files)
psnr_values = []  # 用于存储所有PSNR值，以便后面计算平均值

# 结果保存路径
results_file = os.path.join(output_dir, "quality_evaluation_results.txt")

# 打开结果文件，使用UTF-8编码避免中文乱码
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("图像质量评估结果\n")
    f.write(f"基准目录: {benchmark_dir}\n")
    f.write(f"测试目录: {test_dir}\n")
    f.write("\n")
    f.write("文件名\tSSIM\tPSNR\tLPIPS\n")
    
    # 批量处理参数
    batch_size = 1  # 每次处理1张图片，可根据内存情况调整

    # 分批次处理图片
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        
        print(f"\n处理批次 {batch_start//batch_size + 1}/{(num_images + batch_size - 1)//batch_size} (图片 {batch_start+1} 到 {batch_end})")
        
        for i in range(batch_start, batch_end):
            benchmark_file = benchmark_files[i]
            test_file = test_files[i]
            
            # 确保文件名匹配
            assert benchmark_file == test_file, f"文件名不匹配: {benchmark_file} vs {test_file}"
            
            # 加载图片
            benchmark_img_path = os.path.join(benchmark_dir, benchmark_file)
            test_img_path = os.path.join(test_dir, test_file)
            
            benchmark_img = Image.open(benchmark_img_path).convert('RGB')
            test_img = Image.open(test_img_path).convert('RGB')
            
            # 转换为Tensor
            benchmark_tensor = img_transform(benchmark_img).unsqueeze(0).to(device)
            test_tensor = img_transform(test_img).unsqueeze(0).to(device)
            
            # 计算SSIM
            current_ssim = ssim(benchmark_tensor, test_tensor, window_size=11, size_average=True).item()
            
            # 计算PSNR
            mse = F.mse_loss(benchmark_tensor, test_tensor)
            current_psnr = -10.0 * torch.log10(mse).item()
            psnr_values.append(current_psnr)  # 保存PSNR值
            
            # 计算LPIPS - 移动到CPU进行计算以避免GPU内存不足
            benchmark_tensor_cpu = benchmark_tensor.cpu()
            test_tensor_cpu = test_tensor.cpu()
            
            # 降低分辨率以减少LPIPS计算所需内存
            lpips_resolution = 256  # 降低到256x256
            low_res_benchmark = F.interpolate(benchmark_tensor_cpu, size=(lpips_resolution, lpips_resolution), mode='bilinear', align_corners=False)
            low_res_test = F.interpolate(test_tensor_cpu, size=(lpips_resolution, lpips_resolution), mode='bilinear', align_corners=False)
            
            # 计算LPIPS
            current_lpips = lpips_net(low_res_benchmark, low_res_test).item()
            
            # 清理CPU张量
            del benchmark_tensor_cpu
            del test_tensor_cpu
            del low_res_benchmark
            del low_res_test
            
            # 更新统计
            total_ssim += current_ssim
            total_psnr += current_psnr
            total_lpips += current_lpips
            
            # 写入单张图片结果
            f.write(f"{benchmark_file}\t{current_ssim:.6f}\t{current_psnr:.6f}\t{current_lpips:.6f}\n")
            
            print(f"处理图片 {i+1}/{num_images}: {benchmark_file}")
            print(f"  SSIM: {current_ssim:.6f}")
            print(f"  PSNR: {current_psnr:.6f}")
            print(f"  LPIPS: {current_lpips:.6f}")
            
            # 释放图片资源
            benchmark_img.close()
            test_img.close()
            
            # 清理Tensor
            del benchmark_tensor
            del test_tensor
            
        # 清理CUDA缓存
        torch.cuda.empty_cache()
    
    # 计算平均值，处理PSNR为inf的情况
    avg_ssim = total_ssim / num_images
    
    # 使用psnr_values列表计算平均PSNR
    # 统计非inf的PSNR值
    valid_psnr_values = [psnr for psnr in psnr_values if psnr != float('inf')]
    valid_psnr_count = len(valid_psnr_values)
    
    if valid_psnr_count > 0:
        avg_psnr = sum(valid_psnr_values) / valid_psnr_count
    else:
        avg_psnr = float('inf')
    
    avg_lpips = total_lpips / num_images
    
    # 写入平均值
    f.write("\n")
    f.write(f"平均值\t{avg_ssim:.6f}\t{avg_psnr:.6f}\t{avg_lpips:.6f}\n")

print("\n评估完成!")
print(f"结果保存到: {results_file}")
print(f"平均SSIM: {avg_ssim:.6f}")
print(f"平均PSNR: {avg_psnr:.6f}")
print(f"平均LPIPS: {avg_lpips:.6f}")
