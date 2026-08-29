import os
import json
import numpy as np
from plyfile import PlyData, PlyElement


def load_ply_model(ply_path):
    """
    加载PLY模型并返回顶点数据
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"File not found: {ply_path}")
    
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data
    return vertex_data


def load_replacement_table(json_path):
    """
    加载替换表，返回索引映射关系
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    
    with open(json_path, 'r') as f:
        replacement_table = json.load(f)
    
    # 将键转换为整数
    replacement_table = {int(k): v for k, v in replacement_table.items()}
    return replacement_table


def extract_and_reconstruct_ggbond():
    # 文件路径配置
    new_treehill_path = r"D:\latest\new way\output\80000\new_treehill.ply"
    replacement_table_path = r"D:\latest\new way\output\80000\replacement_table.json"
    ggbond_zero_rot_path = r"D:\latest\new way\output\ggbond_rotation_zero.ply"  # 使用旋转接近零的ggbond模型
    output_ggbond_path = r"D:\latest\new way\output\80000\reconstructed_ggbond.ply"
    
    # 1. 加载替换表
    print("Loading replacement table...")
    replacement_table = load_replacement_table(replacement_table_path)
    print(f"Loaded replacement table with {len(replacement_table)} entries")
    
    # 2. 加载new_treehill.ply模型
    print("Loading new_treehill.ply...")
    new_treehill_vertices = load_ply_model(new_treehill_path)
    print(f"Loaded new_treehill.ply with {len(new_treehill_vertices)} vertices")
    
    # 3. 加载旋转接近零的ggbond模型
    print("Loading ggbond_rotation_zero.ply model...")
    ggbond_zero_rot_vertices = load_ply_model(ggbond_zero_rot_path)
    print(f"Loaded ggbond_rotation_zero.ply model with {len(ggbond_zero_rot_vertices)} vertices")
    
    # 4. 重建ggbond模型
    print("Reconstructing ggbond model...")
    
    # 创建一个与ggbond_rotation_zero模型相同大小的数组
    reconstructed_vertices_array = np.empty_like(ggbond_zero_rot_vertices)
    
    # 首先将ggbond_rotation_zero模型的所有顶点复制到新数组（获取其他属性）
    reconstructed_vertices_array[:] = ggbond_zero_rot_vertices
    
    # 遍历替换表中的每个映射关系
    for low_idx, ggbond_idx in replacement_table.items():
        # 从new_treehill.ply获取使用了ggbond旋转参数的高斯球的信息
        treehill_vertex = new_treehill_vertices[low_idx]
        
        # 更新旋转信息 - 使用new_treehill.ply中的旋转参数
        reconstructed_vertices_array[ggbond_idx]['rot_0'] = treehill_vertex['rot_0']
        reconstructed_vertices_array[ggbond_idx]['rot_1'] = treehill_vertex['rot_1']
        reconstructed_vertices_array[ggbond_idx]['rot_2'] = treehill_vertex['rot_2']
        reconstructed_vertices_array[ggbond_idx]['rot_3'] = treehill_vertex['rot_3']
        
        # 根据用户要求，不需要修改坐标，因为ggbond_rotation_zero.ply已经包含了正确的ggbond模型坐标
        # 颜色、缩放、不透明度、法向量等其他属性也保持ggbond_rotation_zero模型的不变
        # 我们只需要更新旋转参数
    
    # 5. 生成重建的PLY文件
    print("Generating reconstructed ggbond model...")
    
    # 创建新的元素
    reconstructed_element = PlyElement.describe(
        reconstructed_vertices_array,
        'vertex'
    )
    
    # 创建新的PLY数据
    reconstructed_ply_data = PlyData([reconstructed_element], text=False)
    
    # 保存新的PLY文件
    reconstructed_ply_data.write(output_ggbond_path)
    print(f"Reconstructed ggbond model saved to: {output_ggbond_path}")
    print(f"Generated {len(reconstructed_vertices_array)} vertices in the reconstructed model")


if __name__ == "__main__":
    try:
        extract_and_reconstruct_ggbond()
        print("\nProcess completed successfully!")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()