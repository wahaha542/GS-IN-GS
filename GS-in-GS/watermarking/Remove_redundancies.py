import os
import numpy as np
import json
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

def get_unselected_indices(low_contribution_path):
    """获取未被选择的高斯球索引"""
    # 加载低贡献模型
    low_vertices, _ = load_model(low_contribution_path)
    if low_vertices is None:
        return None
    
    total_spheres = len(low_vertices)
    num_selected = 71567
    num_unselected = total_spheres - num_selected
    
    print(f"总共有 {total_spheres} 个低贡献高斯球")
    print(f"其中 {num_selected} 个被选中用于旋转替换")
    print(f"剩余 {num_unselected} 个需要从new_treehill模型中删除")
    
    # 使用相同的随机种子重新生成选中的索引
    np.random.seed(42)  # 必须与replace.py中使用的种子相同
    selected_indices = np.random.choice(total_spheres, num_selected, replace=False)
    
    # 生成所有索引的集合
    all_indices = set(range(total_spheres))
    
    # 计算未被选中的索引
    selected_indices_set = set(selected_indices)
    unselected_indices = list(all_indices - selected_indices_set)
    
    print(f"成功确定 {len(unselected_indices)} 个未被选中的高斯球索引")
    return low_vertices, unselected_indices

def get_unselected_coordinates(low_vertices, unselected_indices):
    """获取未被选中的高斯球的坐标"""
    print(f"正在获取 {len(unselected_indices)} 个未被选中的高斯球的坐标")
    
    coordinates = []
    for idx in unselected_indices:
        vertex = low_vertices[idx]
        coords = (vertex['x'], vertex['y'], vertex['z'])
        coordinates.append(coords)
    
    print(f"成功获取所有未被选中的高斯球坐标")
    return coordinates

def remove_vertices_by_coordinates(new_treehill_path, coordinates, output_path):
    """从new_treehill模型中删除指定坐标的顶点"""
    # 加载new_treehill模型
    new_treehill_vertices, new_treehill_ply = load_model(new_treehill_path)
    if new_treehill_vertices is None:
        return False
    
    print(f"原始new_treehill模型包含 {len(new_treehill_vertices)} 个高斯球")
    print(f"开始删除 {len(coordinates)} 个指定坐标的高斯球")
    
    # 创建坐标到索引的映射，保留足够的小数位数以避免精度问题
    coord_to_index = {}
    for i, vertex in enumerate(new_treehill_vertices):
        # 保留6位小数进行比较
        x_str = f"{vertex['x']:.6f}"
        y_str = f"{vertex['y']:.6f}"
        z_str = f"{vertex['z']:.6f}"
        coord_key = (x_str, y_str, z_str)
        coord_to_index[coord_key] = i
    
    # 标记需要删除的索引
    indices_to_remove = []
    for i, coord in enumerate(coordinates):
        # 保留6位小数进行比较
        x_str = f"{coord[0]:.6f}"
        y_str = f"{coord[1]:.6f}"
        z_str = f"{coord[2]:.6f}"
        coord_key = (x_str, y_str, z_str)
        
        if coord_key in coord_to_index:
            indices_to_remove.append(coord_to_index[coord_key])
            # 每1000个删除点打印一次进度
            if (i + 1) % 1000 == 0:
                print(f"已找到 {i + 1} 个要删除的高斯球...")
        else:
            print(f"警告: 未在new_treehill中找到坐标 {coord} 的高斯球")
    
    print(f"成功找到 {len(indices_to_remove)} 个要删除的高斯球")
    
    # 创建新的顶点数组，排除需要删除的索引
    if len(indices_to_remove) > 0:
        # 将新treehill的顶点数据转换为numpy数组，以便进行高效的索引操作
        vertex_array = np.array(new_treehill_vertices)
        
        # 创建一个掩码，指示哪些顶点需要保留
        mask = np.ones(len(vertex_array), dtype=bool)
        mask[indices_to_remove] = False
        
        # 使用掩码过滤顶点
        new_vertices = vertex_array[mask]
    else:
        new_vertices = new_treehill_vertices
    
    print(f"删除完成后，新模型包含 {len(new_vertices)} 个高斯球")
    
    # 保存新模型
    return save_model(new_vertices, new_treehill_ply, output_path)

def main():
    """主函数"""
    # 文件路径
    low_contribution_path = r"D:\latest\new way\output\low_contribution_spheres.ply"
    new_treehill_path = r"D:\latest\new way\output\new_treehill.ply"
    output_path = r"D:\latest\new way\output\new_treehill-8433.ply"
    
    print("=" * 60)
    print("开始执行删除冗余高斯球操作")
    print("=" * 60)
    
    # 步骤1: 获取未被选中的高斯球索引
    low_vertices, unselected_indices = get_unselected_indices(low_contribution_path)
    if low_vertices is None or unselected_indices is None:
        print("无法获取未被选中的高斯球索引，程序终止")
        return
    
    # 步骤2: 获取未被选中的高斯球的坐标
    unselected_coordinates = get_unselected_coordinates(low_vertices, unselected_indices)
    
    # 步骤3: 从new_treehill模型中删除这些坐标的高斯球
    success = remove_vertices_by_coordinates(new_treehill_path, unselected_coordinates, output_path)
    
    if success:
        print("=" * 60)
        print("删除冗余高斯球操作执行完成")
        print(f"新模型已保存到: {output_path}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("删除冗余高斯球操作失败")
        print("=" * 60)

if __name__ == "__main__":
    main()