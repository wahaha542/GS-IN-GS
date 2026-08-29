import torch
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import torchvision

# 添加项目根目录到Python路径
sys.path.append(r"d:\cvpr25_3D-GSW")

from utils.general_utils import safe_state
from scene.cameras import MiniCam
from gaussian_renderer import render, GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from arguments import PipelineParams

# 设置文件路径
model_path = r"D:\latest\new way\output\replace_cαsr\merged_model.ply"
cameras_path = r"D:\latest\data\treehill\cameras.json"
output_path = r"D:\latest\new way\output\replace_cαsr\evaluate\render"

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 加载相机参数

def load_cameras_from_json(cameras_path):
    with open(cameras_path, 'r') as f:
        cameras_data = json.load(f)
    
    cam_infos = []
    for idx, cam_data in enumerate(cameras_data):
        uid = cam_data['id']
        R = np.array(cam_data['rotation'])
        T = np.array(cam_data['position'])
        fx = cam_data['fx']
        fy = cam_data['fy']
        width = cam_data['width']
        height = cam_data['height']
        
        # 将焦距转换为视场角
        FoVy = focal2fov(fy, height)
        FoVx = focal2fov(fx, width)
        znear, zfar = 0.01, 100.0
        
        # 构建变换矩阵
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        # 创建MiniCam对象
        cam = MiniCam(
            width=width, height=height, fovy=FoVy, fovx=FoVx,
            znear=znear, zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform
        )
        cam.uid = uid  # 添加uid属性以便保存图片时命名
        cam_infos.append(cam)
    return cam_infos

# 主渲染函数
def main():
    # 初始化系统状态
    safe_state(True)
    
    with torch.no_grad():
        # 加载Gaussian模型
        gaussians = GaussianModel(3)  # sh_degree=3
        gaussians.load_ply(model_path)
        
        # 加载相机参数
        cam_infos = load_cameras_from_json(cameras_path)
        
        # 设置渲染参数
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        # 正确初始化PipelineParams
        from argparse import ArgumentParser
        parser = ArgumentParser()
        pipeline_params = PipelineParams(parser)
        pipeline = pipeline_params.extract(parser.parse_args([]))
        pipeline.debug = False
        
        # 渲染每个相机视角
        for idx, cam in enumerate(tqdm(cam_infos, desc="Rendering progress")):
            # 执行渲染
            rendering = render(cam, gaussians, pipeline, background)["render"]
            
            # 保存渲染结果
            save_path = os.path.join(output_path, f'{cam.uid:05d}.png')
            torchvision.utils.save_image(rendering, save_path)

if __name__ == "__main__":
    main()