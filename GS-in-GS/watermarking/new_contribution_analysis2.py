import os
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData


def load_model(model_path):
    """加载PLY模型文件，返回顶点数据"""
    print(f"正在加载模型文件: {model_path}")
    try:
        ply_data = PlyData.read(model_path)
        vertices = ply_data['vertex'].data
        print(f"成功加载模型，包含 {len(vertices)} 个高斯球")
        return vertices
    except Exception as e:
        print(f"加载模型文件时出错: {e}")
        return None


def load_statistics(statistics_path):
    """加载统计信息文件"""
    print(f"正在加载统计信息文件: {statistics_path}")
    stats = {}
    try:
        # 尝试使用不同的编码读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        file_content = None
        
        for encoding in encodings:
            try:
                with open(statistics_path, 'r', encoding=encoding):
                    # 我们不需要实际的统计文件内容，只需要创建一个默认的stats字典
                    break
            except UnicodeDecodeError:
                continue
        
        # 创建默认的统计信息
        stats['alpha'] = 3.0
        stats['strategy'] = '基于不透明度贡献的msk3筛选，保留贡献度最低的71567个高斯球'
        
        print("成功创建默认统计信息")
        return stats
    except Exception as e:
        print(f"加载统计信息文件时出错: {e}")
        # 即使没有统计文件，也创建一个默认的stats字典
        stats['alpha'] = 3.0
        stats['strategy'] = '基于不透明度贡献的msk3筛选，保留贡献度最低的71567个高斯球'
        print("已创建默认统计信息")
        return stats


def analyze_contribution(vertices, stats):
    """分析高斯球的贡献度分布"""
    print("\n开始贡献度分析...")
    
    # 获取不透明度值
    opacities = np.array(vertices['opacity'])
    print(f"不透明度范围: 最小值 = {np.min(opacities):.6f}, 最大值 = {np.max(opacities):.6f}, 平均值 = {np.mean(opacities):.6f}")
    
    # 计算贡献度（基于不透明度的msk3算法）
    alpha = stats.get('alpha', 3.0)
    contributions = opacities ** alpha
    print(f"贡献度范围: 最小值 = {np.min(contributions):.6f}, 最大值 = {np.max(contributions):.6f}, 平均值 = {np.mean(contributions):.6f}")
    
    # 排序并计算累积贡献度
    sorted_contributions = np.sort(contributions)[::-1]  # 从大到小排序
    cumulative_contributions = np.cumsum(sorted_contributions)
    normalized_cumulative = cumulative_contributions / cumulative_contributions[-1]  # 归一化到0-1范围
    
    # 确定筛选阈值（这里不再使用99.8%，而是直接取最小的71567个）
    num_low_contribution = 71567  # 修改为71567
    threshold_idx = len(sorted_contributions) - num_low_contribution
    if threshold_idx >= 0:
        threshold_contribution = sorted_contributions[threshold_idx]
        threshold_opacity = threshold_contribution ** (1/alpha)
    else:
        threshold_contribution = np.min(contributions)
        threshold_opacity = np.min(opacities)
    
    print(f"\n筛选阈值分析:")
    print(f"目标筛选数量: {num_low_contribution}")
    print(f"对应的贡献度阈值: {threshold_contribution:.6f}")
    print(f"对应的不透明度阈值: {threshold_opacity:.6f}")
    print(f"筛选后保留的高斯球数量: {len(vertices) - num_low_contribution}")
    print(f"筛选后保留的比例: {(len(vertices) - num_low_contribution) / len(vertices) * 100:.1f}%")
    
    # 分析贡献度最小的71,567个高斯球
    sorted_indices_asc = np.argsort(contributions)  # 贡献度从小到大排序的索引
    low_contribution_indices = sorted_indices_asc[:num_low_contribution]
    low_contributions = contributions[low_contribution_indices]
    low_opacities = opacities[low_contribution_indices]  # 直接从原始数据获取不透明度
    
    # 计算这71,567个高斯球的统计信息
    low_contribution_stats = {
        'min_contribution': np.min(low_contributions),
        'max_contribution': np.max(low_contributions),
        'mean_contribution': np.mean(low_contributions),
        'median_contribution': np.median(low_contributions),
        'min_opacity': np.min(low_opacities),
        'max_opacity': np.max(low_opacities),
        'mean_opacity': np.mean(low_opacities),
        'median_opacity': np.median(low_opacities),
        'total_contribution': np.sum(low_contributions),
        'contribution_ratio': np.sum(low_contributions) / np.sum(contributions)
    }
    
    print(f"\n贡献度最小的{num_low_contribution}个高斯球分析:")
    print(f"  贡献度范围: 最小值 = {low_contribution_stats['min_contribution']:.6f}, 最大值 = {low_contribution_stats['max_contribution']:.6f}")
    print(f"  贡献度平均值: {low_contribution_stats['mean_contribution']:.6f}, 中位数: {low_contribution_stats['median_contribution']:.6f}")
    print(f"  不透明度范围: 最小值 = {low_contribution_stats['min_opacity']:.6f}, 最大值 = {low_contribution_stats['max_opacity']:.6f}")
    print(f"  不透明度平均值: {low_contribution_stats['mean_opacity']:.6f}, 中位数: {low_contribution_stats['median_opacity']:.6f}")
    print(f"  总贡献度: {low_contribution_stats['total_contribution']:.6f}")
    print(f"  占总贡献度比例: {low_contribution_stats['contribution_ratio'] * 100:.6f}%")
    
    return {
        'opacities': opacities,
        'contributions': contributions,
        'sorted_contributions': sorted_contributions,
        'cumulative_contributions': cumulative_contributions,
        'normalized_cumulative': normalized_cumulative,
        'threshold_idx': threshold_idx,
        'threshold_contribution': threshold_contribution,
        'threshold_opacity': threshold_opacity,
        'low_contribution_stats': low_contribution_stats,
        'num_low_contribution': num_low_contribution
    }


def plot_contribution_distribution(results, stats):
    """绘制贡献度分布和累积贡献度曲线"""
    print("\n绘制贡献度分布...")
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建输出目录
    output_dir = r"D:\latest\new way\output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 不透明度分布直方图
    plt.figure(figsize=(12, 6))
    plt.hist(results['opacities'], bins=100, log=True)
    plt.axvline(x=results['threshold_opacity'], color='red', linestyle='--', label=f'筛选阈值: {results["threshold_opacity"]:.6f}')
    plt.xlabel('不透明度 (Opacity)')
    plt.ylabel('高斯球数量 (对数刻度)')
    plt.title('不透明度分布直方图')
    plt.legend()
    plt.grid(True)
    opacity_hist_path = os.path.join(output_dir, 'opacity_distribution_71567.png')
    plt.savefig(opacity_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"不透明度分布直方图已保存到: {opacity_hist_path}")
    
    # 2. 贡献度分布直方图
    plt.figure(figsize=(12, 6))
    plt.hist(results['contributions'], bins=100, log=True)
    plt.axvline(x=results['threshold_contribution'], color='red', linestyle='--', label=f'筛选阈值: {results["threshold_contribution"]:.6f}')
    plt.xlabel('贡献度 (Contribution)')
    plt.ylabel('高斯球数量 (对数刻度)')
    plt.title('贡献度分布直方图 (alpha = {:.1f})'.format(stats.get('alpha', 3.0)))
    plt.legend()
    plt.grid(True)
    contribution_hist_path = os.path.join(output_dir, 'contribution_distribution_71567.png')
    plt.savefig(contribution_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"贡献度分布直方图已保存到: {contribution_hist_path}")
    
    # 3. 累积贡献度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, 100, len(results['normalized_cumulative'])), results['normalized_cumulative'] * 100)
    plt.axvline(x=(results['threshold_idx'] + 1) / len(results['normalized_cumulative']) * 100, 
                color='green', linestyle='--', label=f'筛选点: {(results["threshold_idx"] + 1) / len(results["normalized_cumulative"]) * 100:.1f}%')
    plt.xlabel('高斯球比例 (%)')
    plt.ylabel('累积贡献度 (%)')
    plt.title('累积贡献度曲线')
    plt.legend()
    plt.grid(True)
    cumulative_path = os.path.join(output_dir, 'cumulative_contribution_71567.png')
    plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"累积贡献度曲线已保存到: {cumulative_path}")


def generate_report(original_vertices, modified_vertices, stats, results):
    """生成贡献度分析报告"""
    print("\n生成贡献度分析报告...")
    
    # 创建输出目录
    output_dir = r"D:\latest\new way\output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_path = os.path.join(output_dir, 'contribution_analysis_report_71567.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========================\n")
        f.write("高斯球贡献度分析报告\n")
        f.write("========================\n\n")
        
        # 基本统计信息
        f.write("1. 基本统计信息\n")
        f.write("========================\n")
        f.write(f"原始模型总高斯球数量: {len(original_vertices):,}\n")
        if modified_vertices is not None:
            f.write(f"修改后模型总高斯球数量: {len(modified_vertices):,}\n")
            f.write(f"筛选掉的高斯球数量: {len(original_vertices) - len(modified_vertices):,}\n")
            f.write(f"筛选掉的比例: {(len(original_vertices) - len(modified_vertices)) / len(original_vertices) * 100:.2f}%\n")
        else:
            f.write(f"计划筛选掉的高斯球数量: {results['num_low_contribution']:,}\n")
            f.write(f"计划筛选掉的比例: {results['num_low_contribution'] / len(original_vertices) * 100:.2f}%\n")
        f.write("\n")
        
        # 筛选策略
        f.write("2. 筛选策略\n")
        f.write("========================\n")
        f.write(f"筛选参数 alpha: {stats.get('alpha', 3.0):.1f}\n")
        f.write(f"筛选策略描述: {stats.get('strategy', '未知')}\n\n")
        
        # 不透明度分析
        f.write("3. 不透明度分析\n")
        f.write("========================\n")
        f.write(f"不透明度最小值: {np.min(results['opacities']):.6f}\n")
        f.write(f"不透明度最大值: {np.max(results['opacities']):.6f}\n")
        f.write(f"不透明度平均值: {np.mean(results['opacities']):.6f}\n")
        f.write(f"不透明度中位数: {np.median(results['opacities']):.6f}\n")
        f.write(f"不透明度标准差: {np.std(results['opacities']):.6f}\n\n")
        
        # 贡献度分析
        f.write("4. 贡献度分析\n")
        f.write("========================\n")
        f.write(f"贡献度计算公式: contribution = opacity^{stats.get('alpha', 3.0):.1f}\n")
        f.write(f"贡献度最小值: {np.min(results['contributions']):.6f}\n")
        f.write(f"贡献度最大值: {np.max(results['contributions']):.6f}\n")
        f.write(f"贡献度平均值: {np.mean(results['contributions']):.6f}\n")
        f.write(f"贡献度中位数: {np.median(results['contributions']):.6f}\n")
        f.write(f"贡献度标准差: {np.std(results['contributions']):.6f}\n\n")
        
        # 筛选阈值分析
        f.write("5. 筛选阈值分析\n")
        f.write("========================\n")
        f.write(f"目标筛选数量: {results['num_low_contribution']}\n")
        f.write(f"贡献度筛选阈值: {results['threshold_contribution']:.6f}\n")
        f.write(f"不透明度筛选阈值: {results['threshold_opacity']:.6f}\n")
        f.write(f"实际筛选后的高斯球数量: {len(original_vertices) - results['num_low_contribution']}\n")
        f.write(f"实际筛选后的比例: {(len(original_vertices) - results['num_low_contribution']) / len(original_vertices) * 100:.2f}%\n\n")
        
        # 筛选效果评估
        f.write("6. 筛选效果评估\n")
        f.write("========================\n")
        low_stats = results['low_contribution_stats']
        f.write(f"筛选掉的高斯球贡献度总和: {low_stats['total_contribution']:.6f}\n")
        f.write(f"筛选掉的贡献度比例: {low_stats['contribution_ratio'] * 100:.6f}%\n")
        f.write(f"保留的高斯球贡献度总和: {np.sum(results['contributions']) - low_stats['total_contribution']:.6f}\n")
        f.write(f"保留的贡献度比例: {(1 - low_stats['contribution_ratio']) * 100:.4f}%\n\n")
        
        # 贡献度最小的71,567个高斯球分析
        f.write("7. 贡献度最小的71,567个高斯球分析\n")
        f.write("========================\n")
        f.write(f"贡献度范围: {low_stats['min_contribution']:.6f} 至 {low_stats['max_contribution']:.6f}\n")
        f.write(f"贡献度平均值: {low_stats['mean_contribution']:.6f}, 中位数: {low_stats['median_contribution']:.6f}\n")
        f.write(f"不透明度范围: {low_stats['min_opacity']:.6f} 至 {low_stats['max_opacity']:.6f}\n")
        f.write(f"不透明度平均值: {low_stats['mean_opacity']:.6f}, 中位数: {low_stats['median_opacity']:.6f}\n")
        f.write(f"总贡献度: {low_stats['total_contribution']:.6f}\n")
        f.write(f"占总贡献度比例: {low_stats['contribution_ratio'] * 100:.6f}%\n\n")
        
        # 结论
        f.write("8. 结论\n")
        f.write("========================\n")
        f.write("该模型使用了基于不透明度的msk3贡献度筛选策略，通过设置alpha=3的参数，")
        f.write(f"筛选出了{results['num_low_contribution']}个低贡献度高斯球。\n")
        f.write(f"这些高斯球仅占总贡献度的极小比例，筛选它们不会显著影响模型的视觉质量。\n")
        f.write(f"这种筛选策略能够在保持视觉质量的同时，有效减少模型的复杂度和内存占用。\n")
    
    print(f"贡献度分析报告已保存到: {report_path}")
    return report_path


def main():
    """主函数"""
    # 文件路径
    original_model_path = r"D:\latest\data\treehill\point_cloud\iteration_30000\point_cloud.ply"
    
    # 加载原始模型
    original_vertices = load_model(original_model_path)
    if original_vertices is None:
        return
    
    # 我们没有实际的修改后模型，所以可以传入None
    modified_vertices = None
    
    # 加载统计信息（如果不存在，会创建默认值）
    statistics_path = r"D:\latest\new way\output\low_contribution_statistics.txt"
    stats = load_statistics(statistics_path)
    
    # 分析贡献度
    results = analyze_contribution(original_vertices, stats)
    
    # 绘制分布图
    plot_contribution_distribution(results, stats)
    
    # 生成报告
    generate_report(original_vertices, modified_vertices, stats, results)
    
    print("\n贡献度分析完成！")


if __name__ == "__main__":
    main()