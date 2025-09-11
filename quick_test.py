#!/usr/bin/env python3
"""
快速测试脚本 - 处理点云子集来验证参数
"""

import numpy as np
import laspy
import sys
from pathlib import Path

# Add corridor_seg to path
sys.path.insert(0, '.')

from corridor_seg.config import Config
from corridor_seg.main import CorridorSegmenter

def subsample_point_cloud(input_path, output_path, sample_ratio=0.1):
    """
    对点云进行子采样以快速测试
    """
    print(f"加载点云: {input_path}")
    las = laspy.read(input_path)
    
    total_points = len(las.points)
    sample_size = int(total_points * sample_ratio)
    
    # 随机采样
    indices = np.random.choice(total_points, sample_size, replace=False)
    indices = np.sort(indices)  # 保持顺序
    
    print(f"从{total_points:,}个点中采样{sample_size:,}个点 ({sample_ratio*100:.1f}%)")
    
    # 使用原始数据的切片创建新LAS
    points_subset = las.points[indices]
    
    # 复制原始header并更新点数
    header = las.header
    header.point_count = len(indices)
    
    # 创建新的LAS数据
    las_out = laspy.create(point_format=header.point_format, file_version=header.version)
    las_out.header = header
    las_out.points = points_subset
    
    las_out.write(output_path)
    print(f"保存采样点云到: {output_path}")
    
    return output_path

def quick_test():
    """快速测试流程"""
    input_file = "/Users/lambdayin/Code-Projects/maicro_projects/detection/Spatil-Line-Clustering/data/cloud_merged.las"
    sample_file = "./temp_sample.las"
    
    print("=== 快速测试模式 ===")
    print("1. 创建点云子集...")
    
    # 创建10%的子样本
    subsample_point_cloud(input_file, sample_file, sample_ratio=0.1)
    
    print("\\n2. 运行算法...")
    
    # 使用调试配置
    config = Config("debug_config.yaml")
    segmenter = CorridorSegmenter(config)
    
    # 处理子采样数据
    results = segmenter.segment_corridor(sample_file, "./quick_test_output")
    
    print("\\n=== 快速测试结果 ===")
    print(f"处理时间: {results['processing_stats']['total_time']:.2f}s")
    print(f"检测到的电力线: {len(results['power_lines'])}")
    print(f"检测到的塔架: {len(results['towers'])}")
    print(f"电力线点数: {results['processing_stats']['point_counts']['powerlines']}")
    print(f"塔架点数: {results['processing_stats']['point_counts']['towers']}")
    
    # 清理临时文件
    Path(sample_file).unlink(missing_ok=True)
    
    return results

if __name__ == "__main__":
    np.random.seed(42)  # 可重现结果
    results = quick_test()