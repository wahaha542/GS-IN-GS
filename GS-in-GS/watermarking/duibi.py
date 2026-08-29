import os
import numpy as np
from plyfile import PlyData

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

def compare_models(reconstructed_path, original_path, output_dir):
    """
    比较重建模型和原始模型的差异
    """
    # 加载模型
    reconstructed_vertices = load_ply_model(reconstructed_path)
    original_vertices = load_ply_model(original_path)
    
    if reconstructed_vertices is None or original_vertices is None:
        return
    
    # 检查顶点数量是否匹配
    if len(reconstructed_vertices) != len(original_vertices):
        print(f"顶点数量不匹配: {len(reconstructed_vertices)} vs {len(original_vertices)}")
        return
    
    # 获取共同的属性（排除x, y, z）
    common_properties = list(set(reconstructed_vertices.dtype.names) & set(original_vertices.dtype.names) - {'x', 'y', 'z'})
    
    if not common_properties:
        print("没有共同的属性可比较")
        return
    
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

if __name__ == "__main__":
    # 模型路径
    reconstructed_model = r"D:\latest\new way\output\extract_piliang\ggbond_csr.ply"
    original_model = r"D:\latest\data\ggbond\point_cloud\iteration_30000\point_cloud.ply"
    output_directory = r"D:\latest\new way\output\replace_csr\evaluate"
    
    # 执行比较
    compare_models(reconstructed_model, original_model, output_directory)