import os
import numpy as np
from plyfile import PlyData, PlyElement

def load_ply_model(ply_path):
    """
    加载PLY模型并返回顶点数据
    """
    try:
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex'].data
        return vertices
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def save_ply_model(vertices, output_path):
    """
    保存PLY模型
    """
    try:
        element = PlyElement.describe(vertices, 'vertex')
        plydata = PlyData([element], text=False)
        plydata.write(output_path)
        print(f"模型保存成功: {output_path}")
        return True
    except Exception as e:
        print(f"保存模型失败: {e}")
        return False

def load_replacement_table(json_path):
    """
    加载替换表
    """
    import json
    try:
        with open(json_path, 'r') as f:
            replacement_table = json.load(f)
        return {int(k): v for k, v in replacement_table.items()}
    except Exception as e:
        print(f"加载替换表失败: {e}")
        return None

def reconstruct_ggbond_from_attacked_treehill(attacked_treehill_path, replacement_table_path, ggbond_zero_path, output_path):
    """
    从攻击后的treehill模型中提取并重建ggbond模型
    """
    # 加载攻击后的treehill模型
    attacked_treehill_vertices = load_ply_model(attacked_treehill_path)
    if attacked_treehill_vertices is None:
        return False
    
    # 加载替换表
    replacement_table = load_replacement_table(replacement_table_path)
    if replacement_table is None:
        return False
    
    # 加载ggbond_zero模型
    ggbond_zero_vertices = load_ply_model(ggbond_zero_path)
    if ggbond_zero_vertices is None:
        return False
    
    # 重建ggbond模型
    reconstructed_vertices_array = np.empty_like(ggbond_zero_vertices)
    reconstructed_vertices_array[:] = ggbond_zero_vertices
    
    # 遍历替换表，从攻击后的treehill模型中提取信息
    parameters = ['f_dc_0', 'f_dc_1', 'f_dc_2', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    for low_idx, ggbond_idx in replacement_table.items():
        if low_idx < len(attacked_treehill_vertices) and ggbond_idx < len(reconstructed_vertices_array):
            treehill_vertex = attacked_treehill_vertices[low_idx]
            for param in parameters:
                if param in attacked_treehill_vertices.dtype.names and param in reconstructed_vertices_array.dtype.names:
                    reconstructed_vertices_array[ggbond_idx][param] = treehill_vertex[param]
    
    # 保存重建的ggbond模型
    return save_ply_model(reconstructed_vertices_array, output_path)

def compare_models(reconstructed_path, original_path, output_dir):
    """
    比较重建模型和原始模型的差异
    """
    # 加载模型
    reconstructed_vertices = load_ply_model(reconstructed_path)
    original_vertices = load_ply_model(original_path)
    
    if reconstructed_vertices is None or original_vertices is None:
        return None
    
    # 检查顶点数量是否匹配
    if len(reconstructed_vertices) != len(original_vertices):
        print(f"顶点数量不匹配: {len(reconstructed_vertices)} vs {len(original_vertices)}")
        return None
    
    # 获取共同的属性（排除x, y, z）
    common_properties = list(set(reconstructed_vertices.dtype.names) & set(original_vertices.dtype.names) - {'x', 'y', 'z'})
    
    if not common_properties:
        print("没有共同的属性可比较")
        return None
    
    # 计算差异
    total_differences = 0
    total_values = 0
    threshold = 1e-6  # 浮点数比较阈值
    
    for prop in common_properties:
        recon_values = reconstructed_vertices[prop]
        orig_values = original_vertices[prop]
        
        # 计算绝对差异
        differences = np.abs(recon_values - orig_values)
        non_zero_diff_count = np.sum(differences > threshold)
        
        total_differences += non_zero_diff_count
        total_values += len(recon_values)
    
    # 计算准确率
    accuracy = ((total_values - total_differences) / total_values) * 100.0 if total_values > 0 else 100.0
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'model_comparison_results.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"准确率: {accuracy:.2f}%\n")
        f.write(f"总差异数: {total_differences}/{total_values}\n")
        f.write(f"比较的属性: {common_properties}\n")
        f.write(f"重建模型: {reconstructed_path}\n")
        f.write(f"原始模型: {original_path}\n")
    
    print(f"比较完成，结果已保存到: {output_file}")
    print(f"准确率: {accuracy:.2f}%")
    
    return accuracy

# 攻击方法实现
def attack_noise(vertices, noise_level=1.0):
    """
    对模型添加噪声攻击
    """
    attacked_vertices = np.copy(vertices)
    
    # 对xyz坐标添加噪声
    if 'x' in attacked_vertices.dtype.names and 'y' in attacked_vertices.dtype.names and 'z' in attacked_vertices.dtype.names:
        for coord in ['x', 'y', 'z']:
            noise = np.random.normal(0, noise_level, len(attacked_vertices))
            attacked_vertices[coord] += noise
    
    return attacked_vertices

def attack_rotate(vertices):
    """
    对模型进行旋转攻击
    """
    attacked_vertices = np.copy(vertices)
    
    # 生成随机旋转矩阵（与pointnet_utils.py一致）
    rotation_matrix = np.random.randn(3, 3)
    
    # 应用旋转
    if 'x' in attacked_vertices.dtype.names and 'y' in attacked_vertices.dtype.names and 'z' in attacked_vertices.dtype.names:
        xyz = np.array([attacked_vertices['x'], attacked_vertices['y'], attacked_vertices['z']]).T
        rotated_xyz = np.dot(xyz, rotation_matrix)
        attacked_vertices['x'] = rotated_xyz[:, 0]
        attacked_vertices['y'] = rotated_xyz[:, 1]
        attacked_vertices['z'] = rotated_xyz[:, 2]
    
    return attacked_vertices

def attack_translate(vertices, translate_level=1.0):
    """
    对模型进行平移攻击
    """
    attacked_vertices = np.copy(vertices)
    
    # 对xyz坐标添加平移（与pointnet_utils.py一致）
    if 'x' in attacked_vertices.dtype.names and 'y' in attacked_vertices.dtype.names and 'z' in attacked_vertices.dtype.names:
        for coord in ['x', 'y', 'z']:
            noise = np.random.randn(len(attacked_vertices)) * translate_level
            attacked_vertices[coord] += noise
    
    return attacked_vertices

def attack_crop(vertices, crop_ratio=0.1):
    """
    对模型进行裁剪攻击
    """
    # 随机裁剪掉一部分顶点
    crop_size = int(len(vertices) * crop_ratio)
    if crop_size < 1:
        return vertices
    
    # 随机选择要保留的顶点
    keep_indices = np.random.choice(len(vertices), len(vertices) - crop_size, replace=False)
    keep_indices.sort()
    
    return vertices[keep_indices]

def main():
    # 路径配置
    base_path = r"D:\latest\new way\output"
    attacked_treehill_dir = os.path.join(base_path, "replace_csr", "attacked")
    evaluate_dir = os.path.join(base_path, "replace_csr", "evaluate")
    
    # 原始模型路径
    original_treehill_path = os.path.join(base_path, "replace_csr", "merged_model.ply")
    original_ggbond_path = r"D:\latest\data\ggbond\point_cloud\iteration_30000\point_cloud.ply"
    ggbond_zero_path = os.path.join(base_path, "ggbond_zero", "ggbond_csr.ply")
    replacement_table_path = os.path.join(base_path, "replace_csr", "replacement_table.json")
    
    # 攻击方法列表
    attacks = [
        ("noise", attack_noise),
        ("rotate", attack_rotate),
        ("translate", attack_translate),
        ("crop", attack_crop)
    ]
    
    # 加载原始treehill模型
    original_treehill_vertices = load_ply_model(original_treehill_path)
    if original_treehill_vertices is None:
        print("无法加载原始treehill模型")
        return
    
    # 对每个攻击方法进行测试
    for attack_name, attack_func in attacks:
        print(f"\n=== 测试攻击方法: {attack_name} ===")
        
        # 创建攻击结果目录
        attack_dir = os.path.join(attacked_treehill_dir, attack_name)
        os.makedirs(attack_dir, exist_ok=True)
        
        # 执行攻击
        print(f"执行{attack_name}攻击...")
        attacked_vertices = attack_func(original_treehill_vertices)
        
        # 保存攻击后的treehill模型
        attacked_treehill_path = os.path.join(attack_dir, "attacked_treehill.ply")
        if save_ply_model(attacked_vertices, attacked_treehill_path):
            print(f"攻击后的treehill模型已保存: {attacked_treehill_path}")
            
            # 从攻击后的treehill模型中重建ggbond模型
            reconstructed_ggbond_path = os.path.join(attack_dir, "reconstructed_ggbond.ply")
            if reconstruct_ggbond_from_attacked_treehill(attacked_treehill_path, replacement_table_path, ggbond_zero_path, reconstructed_ggbond_path):
                print(f"从攻击后的treehill模型中重建的ggbond模型已保存: {reconstructed_ggbond_path}")
                
                # 与原始ggbond模型比较
                attack_evaluate_dir = os.path.join(evaluate_dir, attack_name)
                accuracy = compare_models(reconstructed_ggbond_path, original_ggbond_path, attack_evaluate_dir)
                if accuracy is not None:
                    print(f"{attack_name}攻击后的准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()