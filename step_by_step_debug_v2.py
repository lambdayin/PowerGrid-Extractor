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
from corridor_seg.towers import TowerExtractor
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
    tower_head_height = preprocessed['tower_head_height']
    # 原点（与预处理一致）
    bounds = preprocessed.get('bounds', {})
    min_coords = bounds.get('min_coords', None)
    grid_origin = None
    voxel_origin = None
    if min_coords is not None and len(min_coords) >= 3:
        grid_origin = (float(min_coords[0]), float(min_coords[1]))
        voxel_origin = (float(min_coords[0]), float(min_coords[1]), float(min_coords[2]))
    
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
            title="Step 3: Voxel Grid Structure",
            grid_origin=grid_origin,
            voxel_origin=voxel_origin
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
    graph = powerline_extractor.build_segment_graph(segments, grid_2d, grid_origin=grid_origin)
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

    pl_mask = np.zeros(len(filtered_points), dtype=bool)
    for pl in power_lines:
        pl_mask[pl['point_indices']] = True
    
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
    
    # 7. Tower Detection Pipeline
    print("\\n=== Step 5: 塔检测流程 ===")
    tower_extractor = TowerExtractor(config)
    
    # 估算塔头高度（基于电力线高度）
    # if filtered_lines:
    #     line_heights = []
    #     for pl in filtered_lines:
    #         if 'point_indices' in pl and len(pl['point_indices']) > 0:
    #             pl_points = filtered_points[pl['point_indices']]
    #             line_heights.extend(pl_points[:, 2])
        
    #     if line_heights:
    #         # 塔头高度估算为电力线高度的95%分位数
    #         tower_head_height = np.percentile(line_heights, 95)
    #         print(f"  基于电力线高度估算塔头高度: {tower_head_height:.2f}m")
    #     else:
    #         tower_head_height = 15.0  # 默认值
    #         print("  使用默认塔头高度: 15.0m")
    # else:
    #     tower_head_height = 15.0
    #     print("  使用默认塔头高度: 15.0m")

    # 需要先计算网格特征（包含HeightDiff）
    print("\\n=== Step 5.0: 计算网格特征（用于塔检测）===")
    
    # grid_features = {}
    
    # for grid_idx, point_indices in grid_2d.items():
    #     if not point_indices:
    #         continue
            
    #     grid_points = filtered_points[point_indices]
    #     heights = grid_points[:, 2]
        
    #     # 计算网格特征
    #     features = {
    #         'point_count': len(point_indices),
    #         'centroid': grid_points.mean(axis=0),
    #         'HeightDiff': heights.max() - heights.min(),
    #         'max_height': heights.max(),
    #         'min_height': heights.min(),
    #         'density': len(point_indices) / (config.grid_2d_size ** 2)
    #     }
    #     grid_features[grid_idx] = features

    grid_features = feature_calc.compute_2d_grid_features(
            grid_2d, filtered_points, pl_candidate_mask=pl_mask)
    grid_features = feature_calc.compute_height_based_features(
            grid_features, delta_h_min, tower_head_height)

    print(f"✅ 计算网格特征: {len(grid_features)} 个网格")
    stats['grid_features'] = len(grid_features)
    
    # Step 5.1: 初始高度差筛选
    candidates_step1 = tower_extractor.step1_height_diff_initial_screening(
        grid_features, delta_h_min, tower_head_height)
    
    print(f"✅ 塔检测步骤1: {len(candidates_step1)} 个候选网格")
    stats['tower_step1_candidates'] = len(candidates_step1)
    
    # 可视化塔检测步骤1
    if visualizer and candidates_step1:
        visualizer.visualize_tower_step1_initial_screening(
            candidates_step1, grid_features, delta_h_min, tower_head_height,
            title="Tower Step 1: Initial Height Screening"
        )
    
    if not candidates_step1:
        print("❌ 塔检测步骤1失败：没有候选网格")
        stats.update({
            'tower_step2_candidates': 0,
            'tower_step3_candidates': 0,
            'tower_clusters': 0,
            'final_towers': 0,
            'tower_points': 0
        })
    else:
        # Step 5.2: 移动窗口细化
        candidates_step2 = tower_extractor.step2_moving_window_refinement(
            grid_features, candidates_step1, tower_head_height, delta_h_min)
        
        print(f"✅ 塔检测步骤2: {len(candidates_step2)} 个候选网格")
        stats['tower_step2_candidates'] = len(candidates_step2)
        
        # 可视化塔检测步骤2
        if visualizer and candidates_step2:
            visualizer.visualize_tower_step2_window_refinement(
                candidates_step1, candidates_step2, grid_features, tower_head_height,
                title="Tower Step 2: Moving Window Refinement"
            )
        
        if not candidates_step2:
            print("❌ 塔检测步骤2失败：没有候选网格")
            stats.update({
                'tower_step3_candidates': 0,
                'tower_clusters': 0,
                'final_towers': 0,
                'tower_points': 0
            })
        else:
            # Step 5.3: 垂直连续性检查
            candidates_step3 = tower_extractor.step3_vertical_continuity_check(
                candidates_step2, grid_features, filtered_points, tower_head_height,
                grid_2d=grid_2d, grid_origin=grid_origin)
            
            print(f"✅ 塔检测步骤3: {len(candidates_step3)} 个候选网格")
            stats['tower_step3_candidates'] = len(candidates_step3)
            
            # 可视化塔检测步骤3
            if visualizer and candidates_step3:
                visualizer.visualize_tower_step3_vertical_continuity(
                    candidates_step2, candidates_step3, grid_features, 
                    filtered_points, tower_head_height,
                    title="Tower Step 3: Vertical Continuity Check"
                )
            
            if not candidates_step3:
                print("❌ 塔检测步骤3失败：没有候选网格")
                stats.update({
                    'tower_clusters': 0,
                    'final_towers': 0,
                    'tower_points': 0
                })
            else:
                # Step 5.4: 聚类成塔
                tower_clusters = tower_extractor.step4_clustering_to_towers(
                    candidates_step3, grid_features)
                
                print(f"✅ 塔检测步骤4: {len(tower_clusters)} 个塔聚类")
                stats['tower_clusters'] = len(tower_clusters)
                
                # 可视化塔检测步骤4
                if visualizer and tower_clusters:
                    visualizer.visualize_tower_step4_clustering(
                        candidates_step3, tower_clusters, grid_features,
                        title="Tower Step 4: Clustering to Towers"
                    )
                
                if not tower_clusters:
                    print("❌ 塔检测步骤4失败：没有塔聚类")
                    stats.update({
                        'final_towers': 0,
                        'tower_points': 0
                    })
                else:
                    # Step 5.5: 平面半径约束
                    final_towers = tower_extractor.step5_planar_radius_constraint(
                        tower_clusters, filtered_points, grid_2d=grid_2d, grid_origin=grid_origin)
                                 
                    print(f"✅ 塔检测步骤5: {len(final_towers)} 个最终塔")
                    stats['final_towers'] = len(final_towers)
                    
                    # 计算塔的点数统计
                    tower_points = sum(len(t.get('points', [])) for t in final_towers)
                    stats['tower_points'] = tower_points
                    
                    # 可视化塔检测步骤5和最终结果
                    if visualizer:
                        visualizer.visualize_tower_step5_radius_constraint(
                            tower_clusters, final_towers, filtered_points,
                            title="Tower Step 5: Final Tower Results"
                        )
                    
                    if final_towers:
                        for i, tower in enumerate(final_towers):
                            points_count = len(tower.get('points', []))
                            radius = tower.get('radius', 0)
                            height_diff = tower.get('max_height_diff', 0)
                            print(f"  塔 {i}: 点数={points_count}, 半径={radius:.2f}m, 高度差={height_diff:.2f}m")
                    
                    # 可视化完整的电力网格系统
                    if visualizer and (filtered_lines or final_towers):
                        visualizer.visualize_complete_power_grid_system(
                            filtered_lines, final_towers, filtered_points,
                            title="Complete Power Grid System (Lines + Towers)"
                        )
    
    # 可视化电力线最终结果
    if visualizer:
        visualizer.visualize_final_results(
            power_lines, filtered_lines, filtered_points,
            title="Step 4: Final Power Line Results"
        )
        
        # 创建总结报告
        visualizer.create_summary_report(stats)
    
    print("\\n=== 调试完成 ===")
    final_towers_count = stats.get('final_towers', 0)
    print(f"最终结果: {len(filtered_lines)} 条电力线 + {final_towers_count} 个塔")
    
    if visualizer:
        print(f"\\n📊 所有可视化结果已保存到: {visualizer.save_dir}")
        print("   包含以下文件:")
        print("   • 00_summary_report.png - 算法执行总结")
        print("   • 01_original_pointcloud.png - 原始点云")
        print("   • 02_preprocessing_results.png - 预处理结果")
        print("   • 03_voxel_grid.png - 体素网格结构")
        print("   • 04_linear_voxels.png - 线性体素分析")
        print("   • 05_power_line_segments.png - 电力线段")
        print("   • 06_final_results.png - 电力线最终结果")
        if stats.get('tower_step1_candidates', 0) > 0:
            print("   • 07_tower_step1_screening.png - 塔检测步骤1")
            print("   • 08_tower_step2_refinement.png - 塔检测步骤2")
            print("   • 09_tower_step3_continuity.png - 塔检测步骤3")
            print("   • 10_tower_step4_clustering.png - 塔检测步骤4")
            print("   • 11_tower_step5_final.png - 塔检测最终结果")
            print("   • 12_complete_system.png - 完整电力网格系统")
    
    return stats

if __name__ == "__main__":
    np.random.seed(42)
    
    # 检查命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='PowerGrid算法逐步调试工具')
    parser.add_argument('--no-vis', action='store_true', help='禁用可视化')
    parser.add_argument('--save-dir', default='debug_visualizations', help='可视化保存目录')
    
    args = parser.parse_args()
    
    enable_visualization = not args.no_vis
    
    stats = step_by_step_debug(enable_visualization, args.save_dir)
    
    print("\\n" + "="*50)
    print("📋 调试会话统计:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print("="*50)