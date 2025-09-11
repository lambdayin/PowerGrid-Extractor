#!/usr/bin/env python3
"""
深度调试分析脚本 - 分析为什么检测不到输电设施
"""

import numpy as np
import laspy
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add corridor_seg to path
sys.path.insert(0, '.')

from corridor_seg.config import Config
from corridor_seg.preprocessing import PointCloudPreprocessor
from corridor_seg.features import FeatureCalculator

def deep_analysis():
    """深度分析数据特征"""
    input_file = "/Users/lambdayin/Code-Projects/maicro_projects/detection/Spatil-Line-Clustering/data/cloud_merged.las"
    
    print("=== 深度数据分析 ===")
    
    # 1. 加载数据
    print("1. 加载点云数据...")
    las = laspy.read(input_file)
    points = np.vstack([las.x, las.y, las.z]).T
    
    print(f"点云数量: {len(points):,}")
    print(f"数据范围: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # 2. 高度分析
    print("\\n2. 详细高度分析...")
    heights = points[:, 2]
    
    # 创建高度直方图
    hist, bins = np.histogram(heights, bins=200)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 找出主要峰值
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist, height=len(points)*0.001, distance=10)
    
    print(f"检测到 {len(peaks)} 个高度峰值:")
    for i, peak_idx in enumerate(peaks):
        peak_height = bin_centers[peak_idx]
        peak_count = hist[peak_idx]
        print(f"  峰值 {i+1}: 高度={peak_height:.2f}m, 点数={peak_count:,}")
    
    # 3. 空间分布分析
    print("\\n3. 空间分布分析...")
    
    # 按高度分层分析
    layers = [
        ("地面层", 0, 10),
        ("低层", 10, 20), 
        ("中层", 20, 35),
        ("高层", 35, 50),
        ("超高层", 50, 100)
    ]
    
    for layer_name, min_h, max_h in layers:
        mask = (heights >= min_h) & (heights < max_h)
        count = np.sum(mask)
        if count > 0:
            layer_points = points[mask]
            print(f"  {layer_name} ({min_h}-{max_h}m): {count:,}点 ({count/len(points)*100:.2f}%)")
            print(f"    高度范围: {layer_points[:, 2].min():.2f}-{layer_points[:, 2].max():.2f}m")
    
    # 4. 使用极宽松参数测试特征提取
    print("\\n4. 测试特征提取...")
    
    # 子采样以加快分析
    sample_size = min(1000000, len(points))  # 最多100万点
    indices = np.random.choice(len(points), sample_size, replace=False)
    sample_points = points[indices]
    print(f"使用 {len(sample_points):,} 点进行特征分析")
    
    # 使用极宽松配置
    config = Config()
    config.grid_2d_size = 10.0
    config.voxel_size = 1.0
    config.a1d_linear_thr = 0.1  # 极低阈值
    
    # 预处理
    preprocessor = PointCloudPreprocessor(config)
    preprocessed = preprocessor.preprocess(sample_points)
    
    # 特征计算
    feature_calc = FeatureCalculator(config)
    voxel_features = feature_calc.compute_voxel_features(
        preprocessed['voxel_hash_3d'], preprocessed['points'])
    
    print(f"体素特征计算完成: {len(voxel_features)} 个体素")
    
    # 分析维度特征分布
    if voxel_features:
        a1ds = [f['a1D'] for f in voxel_features.values()]
        a2ds = [f['a2D'] for f in voxel_features.values()]
        a3ds = [f['a3D'] for f in voxel_features.values()]
        
        print(f"\\n维度特征统计:")
        print(f"  a1D (线性): 平均={np.mean(a1ds):.3f}, 最大={np.max(a1ds):.3f}, >0.5的比例={np.sum(np.array(a1ds)>0.5)/len(a1ds)*100:.1f}%")
        print(f"  a2D (平面): 平均={np.mean(a2ds):.3f}, 最大={np.max(a2ds):.3f}")
        print(f"  a3D (球形): 平均={np.mean(a3ds):.3f}, 最大={np.max(a3ds):.3f}")
        
        # 找出高线性度的体素
        high_linear = [(k, v) for k, v in voxel_features.items() if v['a1D'] > 0.3]
        print(f"\\n高线性度体素 (a1D > 0.3): {len(high_linear)} 个")
        
        if high_linear:
            # 分析这些体素的高度分布
            linear_heights = []
            for k, v in high_linear:
                if 'centroid' in v:
                    linear_heights.append(v['centroid'][2])
            
            if linear_heights:
                print(f"  高线性体素高度范围: {min(linear_heights):.2f}-{max(linear_heights):.2f}m")
                print(f"  高线性体素平均高度: {np.mean(linear_heights):.2f}m")
    
    # 5. 手动检查可能的输电线特征
    print("\\n5. 寻找可能的输电线结构...")
    
    # 按高度分层，寻找线性结构
    for min_height in [5, 10, 15, 20, 25, 30]:
        high_points = points[heights > min_height]
        if len(high_points) < 1000:
            continue
            
        print(f"\\n检查高于{min_height}m的点 ({len(high_points):,}个):")
        
        # 简单的线性检测：计算点在主方向上的分布
        if len(high_points) > 100:
            # 使用PCA找主方向
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(high_points)
            
            explained_variance = pca.explained_variance_ratio_
            print(f"  PCA方差解释比: {explained_variance}")
            
            # 如果第一主成分解释了大部分方差，可能是线性结构
            if explained_variance[0] > 0.8:
                print(f"  *** 发现可能的线性结构！第一主成分解释了{explained_variance[0]*100:.1f}%的方差 ***")
                
                # 计算线性度
                linearity = (explained_variance[0] - explained_variance[1]) / explained_variance[0]
                print(f"  线性度: {linearity:.3f}")

if __name__ == "__main__":
    np.random.seed(42)
    deep_analysis()