#!/usr/bin/env python3
"""
逐步调试 - 跟踪算法每个步骤的输出
带可视化功能，方便调试和分析
"""

import numpy as np
import laspy
import sys
import os

sys.path.insert(0, '.')

from corridor_seg.config import Config
from corridor_seg.preprocessing import PointCloudPreprocessor
from corridor_seg.features import FeatureCalculator
from corridor_seg.powerlines import PowerLineExtractor
from visualization import PowerGridVisualizer

def step_by_step_debug(enable_visualization=True, save_dir="debug_visualizations"):
    """逐步执行算法并检查每步输出
    
    Args:
        enable_visualization: 是否启用可视化
        save_dir: 可视化结果保存目录
    """
    
    print("=== 逐步调试模式（带可视化）===")
    
    # 初始化可视化工具
    visualizer = None
    if enable_visualization:
        visualizer = PowerGridVisualizer(save_dir)
        print(f"📊 可视化结果将保存到: {save_dir}")
    
    # 用于收集统计信息
    stats = {}
    
    # 1. 加载小样本数据
    input_file = "/home/lambdayin/Code-Projects/maicro-projects/detection/3d/Spatil-Line-Clustering/data/cloud4db26d1a9662f7ae_Block_0.las"
    las = laspy.read(input_file)
    sample_points = np.vstack([las.x, las.y, las.z]).T
    
    # 取一个较小的样本进行调试
    # sample_size = 500000  # 50万点
    # indices = np.random.choice(len(points), sample_size, replace=False)
    # sample_points = points[indices]
    
    print(f"使用样本: {len(sample_points):,} 点")
    print(f"高度范围: {sample_points[:, 2].min():.2f} - {sample_points[:, 2].max():.2f}m")
    
    # 收集统计信息
    stats['original_points'] = len(sample_points)
    stats['height_range'] = f"{sample_points[:, 2].min():.2f} - {sample_points[:, 2].max():.2f}m"
    
    # 可视化原始点云
    if visualizer:
        visualizer.visualize_original_pointcloud(
            sample_points, 
            title="Step 1: Original Point Cloud"
        )
    
    # 2. 使用极宽松的配置
    # config = Config()
    # config.grid_2d_size = 5.0
    # config.voxel_size = 0.5
    # config.min_height_gap = 10.0     # 极低
    # config.a1d_linear_thr = 0.6      # 降低
    # config.collinearity_angle_thr = 20.0  # 增大

    config = Config("./examples/custom_config.yaml")
    
    print(f"配置参数:")
    print(f"  min_height_gap: {config.min_height_gap}m")
    print(f"  a1d_linear_thr: {config.a1d_linear_thr}")
    
    # 3. Step 1: 预处理
    print("\\n=== Step 1: 预处理 ===")
    preprocessor = PointCloudPreprocessor(config)
    preprocessed = preprocessor.preprocess(sample_points)
    
    filtered_points = preprocessed['points']
    grid_2d = preprocessed['grid_2d']
    voxel_hash_3d = preprocessed['voxel_hash_3d']
    delta_h_min = preprocessed['delta_h_min']
    
    print(f"✅ 过滤后点数: {len(filtered_points):,}")
    print(f"✅ 2D网格数: {len(grid_2d)}")
    print(f"✅ 3D体素数: {len(voxel_hash_3d)}")
    print(f"✅ Δh_min: {delta_h_min:.2f}m")
    
    # 收集统计信息
    stats['filtered_points'] = len(filtered_points)
    stats['removal_rate'] = (len(sample_points) - len(filtered_points)) / len(sample_points) * 100
    stats['grid_2d_count'] = len(grid_2d)
    stats['voxel_3d_count'] = len(voxel_hash_3d)
    stats['delta_h_min'] = delta_h_min
    stats['voxel_size'] = config.voxel_size
    
    # 可视化预处理结果
    if visualizer:
        visualizer.visualize_preprocessed_points(
            sample_points, filtered_points, delta_h_min,
            title="Step 2: Preprocessing Results"
        )
        
        # 可视化体素网格
        visualizer.visualize_voxel_grid(
            voxel_hash_3d, grid_2d, filtered_points, config,
            title="Step 3: Voxel Grid Structure"
        )
    
    # 4. Step 2: 特征计算
    print("\\n=== Step 2: 特征计算 ===")
    feature_calc = FeatureCalculator(config)
    
    # 计算体素特征
    voxel_features = feature_calc.compute_voxel_features(voxel_hash_3d, filtered_points)
    print(f"✅ 体素特征: {len(voxel_features)}")
    
    # 识别线性结构
    linear_voxels = feature_calc.identify_linear_structures(voxel_features)
    print(f"✅ 线性体素: {len(linear_voxels)}")
    
    # 收集统计信息
    stats['linear_voxels'] = len(linear_voxels)
    stats['a1d_threshold'] = config.a1d_linear_thr
    stats['linear_ratio'] = len(linear_voxels) / len(voxel_features) * 100 if voxel_features else 0
    
    if len(linear_voxels) == 0:
        print("❌ 问题：没有识别到线性结构！")
        
        # 分析为什么没有线性结构
        a1d_values = [f['a1D'] for f in voxel_features.values()]
        print(f"   所有体素的a1D分布:")
        print(f"   平均: {np.mean(a1d_values):.3f}")
        print(f"   最大: {np.max(a1d_values):.3f}")
        print(f"   >0.3的比例: {np.sum(np.array(a1d_values) > 0.3)/len(a1d_values)*100:.1f}%")
        print(f"   >0.6的比例: {np.sum(np.array(a1d_values) > 0.6)/len(a1d_values)*100:.1f}%")
        
        # 手动设置更低的阈值
        print("   尝试更低的阈值...")
        for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
            count = np.sum(np.array(a1d_values) > threshold)
            print(f"   a1D > {threshold}: {count} 体素")
            if count > 0:
                config.a1d_linear_thr = threshold
                linear_voxels = feature_calc.identify_linear_structures(voxel_features)
                print(f"   ✅ 使用阈值{threshold}，识别到{len(linear_voxels)}个线性体素")
                # 更新统计信息
                stats['linear_voxels'] = len(linear_voxels)
                stats['a1d_threshold'] = threshold
                stats['linear_ratio'] = len(linear_voxels) / len(voxel_features) * 100
                break
    
    if len(linear_voxels) == 0:
        print("❌ 仍然没有线性结构，退出调试")
        return
    
    # 5. Step 3: 检查线性体素的高度分布
    print("\\n=== Step 3: 线性体素分析 ===")
    linear_heights = []
    for features in linear_voxels.values():
        if 'centroid' in features:
            linear_heights.append(features['centroid'][2])
    
    print(f"线性体素高度范围: {min(linear_heights):.2f} - {max(linear_heights):.2f}m")
    print(f"平均高度: {np.mean(linear_heights):.2f}m")
    
    # 高度过滤检查
    above_threshold = [h for h in linear_heights if h > delta_h_min]
    print(f"高于Δh_min({delta_h_min:.2f}m)的线性体素: {len(above_threshold)}")
    
    if len(above_threshold) == 0:
        print("❌ 问题：所有线性体素都被高度阈值过滤掉了！")
        print(f"   建议将Δh_min降低到: {min(linear_heights) - 1:.2f}m")
        
        # 即使有问题也要可视化，帮助调试
        if visualizer:
            visualizer.visualize_linear_voxels(
                linear_voxels, voxel_features, voxel_hash_3d, config, delta_h_min,
                title="Step 4: Linear Voxel Analysis (PROBLEMATIC)"
            )
        return
    
    # 可视化线性体素分析
    if visualizer:
        visualizer.visualize_linear_voxels(
            linear_voxels, voxel_features, voxel_hash_3d, config, delta_h_min,
            title="Step 4: Linear Voxel Analysis"
        )
    
    # 6. Step 4: 电力线提取
    print("\\n=== Step 4: 电力线提取 ===")
    powerline_extractor = PowerLineExtractor(config)
    
    # 提取局部段
    segments = powerline_extractor.extract_local_segments(
        linear_voxels, voxel_features, filtered_points, delta_h_min)
    
    print(f"✅ 局部段数: {len(segments)}")
    
    # 收集统计信息
    stats['segments'] = len(segments)
    
    if len(segments) == 0:
        print("❌ 问题：没有提取到局部段！")
        return
    
    # 可视化电力线段
    if visualizer:
        visualizer.visualize_power_line_segments(
            segments, filtered_points,
            title="Step 5: Power Line Segments"
        )
    
    # 分析段的特征
    segment_lengths = [seg.get('length', 0) for seg in segments]
    print(f"段长度范围: {min(segment_lengths):.2f} - {max(segment_lengths):.2f}m")
    print(f"平均段长度: {np.mean(segment_lengths):.2f}m")
    
    # 构建兼容性图
    graph = powerline_extractor.build_segment_graph(segments, grid_2d)
    print(f"✅ 兼容性图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    
    # 收集统计信息
    stats['graph_nodes'] = graph.number_of_nodes()
    stats['graph_edges'] = graph.number_of_edges()
    
    # 全局合并
    power_lines = powerline_extractor.merge_segments_global(graph, segments)
    print(f"✅ 合并后电力线: {len(power_lines)}")
    
    # 收集统计信息
    stats['raw_powerlines'] = len(power_lines)
    
    if len(power_lines) > 0:
        for i, pl in enumerate(power_lines):
            print(f"  电力线 {i}: 长度={pl.get('total_length', 0):.2f}m, "
                  f"段数={pl.get('num_segments', 0)}, 点数={len(pl.get('point_indices', []))}")
    
    # 过滤
    filtered_lines = powerline_extractor.filter_power_lines(power_lines, min_length=10.0, min_segments=1)
    print(f"✅ 过滤后电力线: {len(filtered_lines)}")
    
    # 收集统计信息
    stats['final_powerlines'] = len(filtered_lines)
    if filtered_lines:
        total_length = sum(pl.get('total_length', 0) for pl in filtered_lines)
        stats['total_length'] = total_length
        
        # 计算点数统计
        stats['linear_points'] = sum(len(lv.get('point_indices', [])) for lv in linear_voxels.values() if 'point_indices' in lv)
        stats['segment_points'] = sum(len(seg.get('point_indices', [])) for seg in segments)
        stats['final_points'] = sum(len(pl.get('point_indices', [])) for pl in filtered_lines)
    else:
        stats['total_length'] = 0
        stats['linear_points'] = 0
        stats['segment_points'] = 0
        stats['final_points'] = 0
    
    # 可视化最终结果
    if visualizer:
        visualizer.visualize_final_results(
            power_lines, filtered_lines, filtered_points,
            title="Step 6: Final Power Line Results"
        )
        
        # 创建总结报告
        visualizer.create_summary_report(stats)
    
    print("\\n=== 调试完成 ===")
    print(f"最终结果: {len(filtered_lines)} 条电力线")
    
    if visualizer:
        print(f"\\n📊 所有可视化结果已保存到: {visualizer.save_dir}")
        print("   包含以下文件:")
        print("   • 00_summary_report.png - 算法执行总结")
        print("   • 01_original_pointcloud.png - 原始点云")
        print("   • 02_preprocessing_results.png - 预处理结果")
        print("   • 03_voxel_grid.png - 体素网格结构")
        print("   • 04_linear_voxels.png - 线性体素分析")
        print("   • 05_power_line_segments.png - 电力线段")
        print("   • 06_final_results.png - 最终结果")
    
    return stats

def run_debug_with_config(enable_vis=True, save_dir="debug_visualizations", 
                         voxel_size=0.5, a1d_threshold=0.3, min_height_gap=1.0):
    """运行调试，允许自定义参数
    
    Args:
        enable_vis: 是否启用可视化
        save_dir: 可视化保存目录
        voxel_size: 体素大小
        a1d_threshold: 线性度阈值
        min_height_gap: 最小高度间隙
    """
    print(f"\\n🔧 自定义参数调试模式")
    print(f"   体素大小: {voxel_size}m")
    print(f"   线性度阈值: {a1d_threshold}")
    print(f"   最小高度间隙: {min_height_gap}m")
    
    # 这里可以添加自定义配置的逻辑
    return step_by_step_debug(enable_vis, save_dir)

if __name__ == "__main__":
    np.random.seed(42)
    
    # 检查命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='PowerGrid算法逐步调试工具')
    parser.add_argument('--no-vis', action='store_true', help='禁用可视化')
    parser.add_argument('--save-dir', default='debug_visualizations', help='可视化保存目录')
    parser.add_argument('--quick', action='store_true', help='快速模式（自定义参数）')
    
    args = parser.parse_args()
    
    enable_visualization = not args.no_vis
    
    if args.quick:
        # 快速调试模式，使用更宽松的参数
        stats = run_debug_with_config(
            enable_vis=enable_visualization,
            save_dir=args.save_dir,
            voxel_size=1.0,  # 更大的体素
            a1d_threshold=0.2,  # 更低的阈值
            min_height_gap=0.5  # 更低的高度要求
        )
    else:
        # 标准调试模式
        stats = step_by_step_debug(enable_visualization, args.save_dir)
    
    print("\\n" + "="*50)
    print("📋 调试会话统计:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print("="*50)