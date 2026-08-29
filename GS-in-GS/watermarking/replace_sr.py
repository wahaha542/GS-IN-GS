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

def select_gaussian_spheres(low_contribution_vertices, num_select):
    """从低贡献高斯球中随机均匀选择指定数量的高斯球"""
    print(f"从 {len(low_contribution_vertices)} 个低贡献高斯球中随机选择 {num_select} 个...")
    
    # 生成随机索引
    np.random.seed(42)  # 设置随机种子，确保结果可重复
    selected_indices = np.random.choice(len(low_contribution_vertices), num_select, replace=False)
    
    print(f"成功选择 {len(selected_indices)} 个高斯球")
    return selected_indices

def generate_replacement_table(low_indices, num_ggbond):
    """生成低贡献高斯球与ggbond高斯球的一一对应替换表"""
    print("生成一一对应替换表...")
    
    # 为ggbond高斯球生成随机顺序，确保均匀分布
    np.random.seed(42)
    ggbond_indices = np.random.permutation(num_ggbond)
    
    # 创建替换表
    replacement_table = {}
    for i, low_idx in enumerate(low_indices):
        replacement_table[low_idx] = ggbond_indices[i]
    
    print(f"成功生成 {len(replacement_table)} 个对应关系")
    return replacement_table

def save_replacement_table(replacement_table, output_path):
    """保存替换表为JSON文件"""
    print(f"正在保存替换表到: {output_path}")
    try:
        # 将NumPy整数转换为Python原生整数
        py_replacement_table = {int(k): int(v) for k, v in replacement_table.items()}
        with open(output_path, 'w') as f:
            json.dump(py_replacement_table, f, indent=4)
        print("替换表保存成功")
        return True
    except Exception as e:
        print(f"保存替换表时出错: {e}")
        return False

def replace_scale_rotation_info(low_contribution_vertices, ggbond_vertices, replacement_table):
    """将ggbond高斯球的缩放和旋转信息替换到低贡献高斯球中"""
    print("开始替换缩放和旋转信息...")
    
    # 创建新的顶点数组，复制原始低贡献高斯球的数据
    new_vertices = np.copy(low_contribution_vertices)
    
    # 替换缩放和旋转信息
    for low_idx, ggbond_idx in replacement_table.items():
        # 更新缩放属性
        new_vertices[low_idx]['scale_0'] = ggbond_vertices[ggbond_idx]['scale_0']
        new_vertices[low_idx]['scale_1'] = ggbond_vertices[ggbond_idx]['scale_1']
        new_vertices[low_idx]['scale_2'] = ggbond_vertices[ggbond_idx]['scale_2']
        
        # 更新旋转属性
        new_vertices[low_idx]['rot_0'] = ggbond_vertices[ggbond_idx]['rot_0']
        new_vertices[low_idx]['rot_1'] = ggbond_vertices[ggbond_idx]['rot_1']
        new_vertices[low_idx]['rot_2'] = ggbond_vertices[ggbond_idx]['rot_2']
        new_vertices[low_idx]['rot_3'] = ggbond_vertices[ggbond_idx]['rot_3']
    
    print("缩放和旋转信息替换完成")
    return new_vertices

def main():
    """主函数"""
    # 文件路径
    low_contribution_path = r"D:\latest\new way\output\71567\low_contribution_spheres.ply"
    ggbond_path = r"D:\latest\data\ggbond\point_cloud\iteration_30000\point_cloud.ply"
    output_path = r"D:\latest\new way\output\replace_sr\low_contribution_spheres_updated_scale_rotation.ply"
    replacement_table_path = r"D:\latest\new way\output\replace_sr\replacement_table_scale_rotation.json"
    
    # 加载低贡献模型
    low_contribution_vertices, low_ply_data = load_model(low_contribution_path)
    if low_contribution_vertices is None or low_ply_data is None:
        return
    
    # 加载ggbond模型
    ggbond_vertices, ggbond_ply_data = load_model(ggbond_path)
    if ggbond_vertices is None:
        return
    
    # 验证ggbond模型的高斯球数量
    num_ggbond = len(ggbond_vertices)
    if num_ggbond != 71567:
        print(f"警告: ggbond模型包含 {num_ggbond} 个高斯球，预期应为71567个")
        return
    
    # 选择低贡献高斯球
    selected_indices = select_gaussian_spheres(low_contribution_vertices, num_ggbond)
    
    # 生成替换表
    replacement_table = generate_replacement_table(selected_indices, num_ggbond)
    
    # 保存替换表
    save_replacement_table(replacement_table, replacement_table_path)
    
    # 替换缩放和旋转信息
    new_low_contribution_vertices = replace_scale_rotation_info(
        low_contribution_vertices, ggbond_vertices, replacement_table
    )
    
    # 保存新的低贡献模型
    save_model(new_low_contribution_vertices, low_ply_data, output_path)
    
    print("\n缩放和旋转信息替换完成！")
    print(f"更新后的低贡献高斯球模型已保存到: {output_path}")
    print(f"替换表已保存到: {replacement_table_path}")

if __name__ == "__main__":
    main()
