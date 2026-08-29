import os
import numpy as np
from plyfile import PlyData, PlyElement

def load_model(model_path):
    """加载PLY模型文件，返回顶点数据和原始PLY数据"""
    print(f"正在加载模型文件: {model_path}")
    try:
        ply_data = PlyData.read(model_path)
        vertices = ply_data['vertex'].data
        print(f"成功加载模型，包含 {len(vertices)} 个高斯球")
        return vertices, ply_data
    except Exception as e:
        print(f"加载模型文件时出错: {e}")
        return None, None

def modify_properties(vertices, properties, value):
    """将指定属性设置为指定值"""
    print(f"正在将属性 {properties} 设置为 {value}...")
    
    # 创建新的顶点数组，复制原始数据
    new_vertices = np.copy(vertices)
    
    # 设置指定属性为指定值
    for prop in properties:
        new_vertices[prop] = value
    
    print("属性设置完成")
    return new_vertices

def save_model(vertices, original_ply_data, output_path):
    """保存高斯球模型为PLY文件"""
    print(f"正在保存模型到: {output_path}")
    try:
        # 创建新的顶点元素
        vertex_element = PlyElement.describe(vertices, 'vertex')
        
        # 创建新的PLY数据
        new_ply_data = PlyData([vertex_element], text=original_ply_data.text)
        
        # 保存文件
        new_ply_data.write(output_path)
        print(f"成功保存模型，包含 {len(vertices)} 个高斯球")
        return True
    except Exception as e:
        print(f"保存模型文件时出错: {e}")
        return False

def main():
    """主函数"""
    # 文件路径
    input_path = r"D:\latest\data\ggbond\point_cloud\iteration_30000\point_cloud.ply"
    output_dir = r"D:\latest\new way\output"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载模型
    vertices, ply_data = load_model(input_path)
    if vertices is None or ply_data is None:
        return
    
    # 接近零的极小值
    small_value = 1e-12
    
    # 1. 生成颜色接近零的模型
    color_properties = ['r', 'g', 'b']
    color_output_path = os.path.join(output_dir, "ggbond_color_zero.ply")
    color_vertices = modify_properties(vertices, color_properties, small_value)
    save_model(color_vertices, ply_data, color_output_path)
    
    print("\n" + "="*50 + "\n")
    
    # 2. 生成不透明度接近零的模型
    opacity_properties = ['opacity']
    opacity_output_path = os.path.join(output_dir, "ggbond_opacity_zero.ply")
    opacity_vertices = modify_properties(vertices, opacity_properties, small_value)
    save_model(opacity_vertices, ply_data, opacity_output_path)
    
    print("\n" + "="*50 + "\n")
    
    # 3. 生成缩放接近零的模型
    scale_properties = ['scale_0', 'scale_1', 'scale_2']
    scale_output_path = os.path.join(output_dir, "ggbond_scale_zero.ply")
    scale_vertices = modify_properties(vertices, scale_properties, small_value)
    save_model(scale_vertices, ply_data, scale_output_path)
    
    print("\n" + "="*50 + "\n")
    
    # 4. 生成旋转接近零的模型
    rotation_properties = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    rotation_output_path = os.path.join(output_dir, "ggbond_rotation_zero.ply")
    rotation_vertices = modify_properties(vertices, rotation_properties, small_value)
    save_model(rotation_vertices, ply_data, rotation_output_path)
    
    print("\n所有模型生成完成！")
    print(f"颜色接近零的模型: {color_output_path}")
    print(f"不透明度接近零的模型: {opacity_output_path}")
    print(f"缩放接近零的模型: {scale_output_path}")
    print(f"旋转接近零的模型: {rotation_output_path}")

if __name__ == "__main__":
    main()