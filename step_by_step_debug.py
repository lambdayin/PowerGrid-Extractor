#!/usr/bin/env python3
"""
逐步调试 - 跟踪算法每个步骤的输出
"""

import numpy as np
import laspy
import sys

sys.path.insert(0, '.')

from corridor_seg.config import Config
from corridor_seg.preprocessing import PointCloudPreprocessor
from corridor_seg.features import FeatureCalculator
from corridor_seg.powerlines import PowerLineExtractor

def step_by_step_debug():
    """逐步执行算法并检查每步输出"""
    
    print("=== 逐步调试模式 ===")
    
    # 1. 加载小样本数据
    input_file = "/Users/lambdayin/Code-Projects/maicro_projects/detection/Spatil-Line-Clustering/data/cloud_merged.las"
    las = laspy.read(input_file)
    points = np.vstack([las.x, las.y, las.z]).T
    
    # 取一个较小的样本进行调试
    sample_size = 500000  # 50万点
    indices = np.random.choice(len(points), sample_size, replace=False)
    sample_points = points[indices]
    
    print(f"使用样本: {len(sample_points):,} 点")
    print(f"高度范围: {sample_points[:, 2].min():.2f} - {sample_points[:, 2].max():.2f}m")
    
    # 2. 使用极宽松的配置
    config = Config()
    config.grid_2d_size = 5.0
    config.voxel_size = 0.5
    config.min_height_gap = 1.0      # 极低
    config.a1d_linear_thr = 0.3      # 降低
    config.collinearity_angle_thr = 20.0  # 增大
    
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
    
    # 4. Step 2: 特征计算
    print("\\n=== Step 2: 特征计算 ===")
    feature_calc = FeatureCalculator(config)
    
    # 计算体素特征
    voxel_features = feature_calc.compute_voxel_features(voxel_hash_3d, filtered_points)
    print(f"✅ 体素特征: {len(voxel_features)}")
    
    # 识别线性结构
    linear_voxels = feature_calc.identify_linear_structures(voxel_features)
    print(f"✅ 线性体素: {len(linear_voxels)}")
    
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
                break
    
    if len(linear_voxels) == 0:
        print("❌ 仍然没有线性结构，退出调试")
        return
    
    # 5. Step 3: 检查线性体素的高度分布
    print("\\n=== Step 3: 线性体素分析 ===")
    linear_heights = []
    for voxel_idx, features in linear_voxels.items():
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
        return
    
    # 6. Step 4: 电力线提取
    print("\\n=== Step 4: 电力线提取 ===")
    powerline_extractor = PowerLineExtractor(config)
    
    # 提取局部段
    segments = powerline_extractor.extract_local_segments(
        linear_voxels, voxel_features, filtered_points, delta_h_min)
    
    print(f"✅ 局部段数: {len(segments)}")
    
    if len(segments) == 0:
        print("❌ 问题：没有提取到局部段！")
        return
    
    # 分析段的特征
    segment_lengths = [seg.get('length', 0) for seg in segments]
    print(f"段长度范围: {min(segment_lengths):.2f} - {max(segment_lengths):.2f}m")
    print(f"平均段长度: {np.mean(segment_lengths):.2f}m")
    
    # 构建兼容性图
    graph = powerline_extractor.build_segment_graph(segments, grid_2d)
    print(f"✅ 兼容性图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    
    # 全局合并
    power_lines = powerline_extractor.merge_segments_global(graph, segments)
    print(f"✅ 合并后电力线: {len(power_lines)}")
    
    if len(power_lines) > 0:
        for i, pl in enumerate(power_lines):
            print(f"  电力线 {i}: 长度={pl.get('total_length', 0):.2f}m, "
                  f"段数={pl.get('num_segments', 0)}, 点数={len(pl.get('point_indices', []))}")
    
    # 过滤
    filtered_lines = powerline_extractor.filter_power_lines(power_lines, min_length=10.0, min_segments=1)
    print(f"✅ 过滤后电力线: {len(filtered_lines)}")
    
    print("\\n=== 调试完成 ===")
    print(f"最终结果: {len(filtered_lines)} 条电力线")

if __name__ == "__main__":
    np.random.seed(42)
    step_by_step_debug()