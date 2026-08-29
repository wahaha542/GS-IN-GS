import numpy as np
from plyfile import PlyData, PlyElement
import os


def load_model(model_path):
    """
    加载PLY模型文件
    :param model_path: PLY文件路径
    :return: 顶点数据和原始PLY数据
    """
    print(f'正在加载模型: {model_path}')
    ply_data = PlyData.read(model_path)
    vertex_data = ply_data['vertex'].data
    print(f'模型加载完成，顶点数量: {len(vertex_data)}')
    return vertex_data, ply_data


def merge_vertices(low_vertices, high_vertices):
    """
    合并两个模型的顶点数据
    :param low_vertices: 更新后的低贡献高斯球顶点数据
    :param high_vertices: 高贡献高斯球顶点数据
    :return: 合并后的顶点数据
    """
    print('正在合并顶点数据...')
    
    # 获取顶点属性名
    low_fields = low_vertices.dtype.names
    high_fields = high_vertices.dtype.names
    
    # 确保两个模型的顶点属性一致
    if low_fields != high_fields:
        print(f'警告: 两个模型的顶点属性不一致！')
        print(f'低贡献模型属性: {low_fields}')
        print(f'高贡献模型属性: {high_fields}')
        # 使用低贡献模型的属性
        fields = low_fields
    else:
        fields = low_fields
        print(f'顶点属性一致，使用属性: {fields}')
    
    # 创建合并后的顶点数据数组
    total_vertices = len(low_vertices) + len(high_vertices)
    merged_vertices = np.empty(total_vertices, dtype=low_vertices.dtype)
    
    # 复制低贡献模型的顶点数据
    merged_vertices[:len(low_vertices)] = low_vertices
    
    # 复制高贡献模型的顶点数据
    merged_vertices[len(low_vertices):] = high_vertices
    
    print(f'顶点数据合并完成，总顶点数量: {total_vertices}')
    return merged_vertices


def save_merged_model(merged_vertices, original_ply_data, output_path):
    """
    保存合并后的模型
    :param merged_vertices: 合并后的顶点数据
    :param original_ply_data: 原始PLY数据（用于获取文件格式和属性信息）
    :param output_path: 输出文件路径
    """
    print(f'正在保存合并后的模型到: {output_path}')
    
    # 创建新的PLY元素
    vertex_element = PlyElement.describe(merged_vertices, 'vertex')
    
    # 创建新的PLY数据
    new_ply_data = PlyData([vertex_element], text=original_ply_data.text)
    
    # 保存文件
    new_ply_data.write(output_path)
    print(f'模型保存完成！')


def main():
    # 模型路径
    low_contribution_path = r'D:\latest\new way\output\replace_cαr\low_contribution_spheres_updated_color_opacity_rotation.ply'
    high_contribution_path = r'D:\latest\new way\output\71567\high_contribution_spheres.ply'
    output_path = r'D:\latest\new way\output\replace_cαr\new_treehill_updated_color_opacity_rotation.ply'
    
    # 加载模型
    low_vertices, low_ply_data = load_model(low_contribution_path)
    high_vertices, high_ply_data = load_model(high_contribution_path)
    
    # 合并顶点数据
    merged_vertices = merge_vertices(low_vertices, high_vertices)
    
    # 保存合并后的模型
    save_merged_model(merged_vertices, low_ply_data, output_path)
    
    print('\n=== 模型融合完成 ===')
    print(f'更新后的低贡献高斯球数量: {len(low_vertices)}')
    print(f'高贡献高斯球数量: {len(high_vertices)}')
    print(f'新模型总高斯球数量: {len(merged_vertices)}')
    print(f'新模型文件路径: {output_path}')


if __name__ == '__main__':
    main()
