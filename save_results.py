#!/usr/bin/env python3
"""
结果保存脚本 - 将检测到的电力线保存为不同格式
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from corridor_seg.config import Config
from corridor_seg.main import CorridorSegmenter

def save_detection_results():
    """运行检测并保存结果为多种格式"""
    print("=== 电力线检测和结果保存 ===")
    
    # 使用调试配置
    config = Config("debug_config.yaml")
    segmenter = CorridorSegmenter(config)
    
    # 运行检测
    input_file = "/Users/lambdayin/Code-Projects/maicro_projects/detection/Spatil-Line-Clustering/data/cloud_merged.las"
    print(f"处理文件: {input_file}")
    
    try:
        results = segmenter.process_point_cloud(segmenter.load_point_cloud(input_file))
        
        # 获取结果
        pl_points = results['pl_points']
        tower_points = results['tower_points']
        power_lines = results['power_lines']
        
        print(f"\\n检测结果:")
        print(f"电力线: {len(power_lines)} 条")
        print(f"电力线点: {len(pl_points)} 个")
        print(f"塔架点: {len(tower_points)} 个")
        
        # 保存为多种格式
        output_dir = "./detection_results"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存为numpy格式
        if len(pl_points) > 0:
            np.save(f"{output_dir}/powerlines_points.npy", pl_points)
            print(f"✅ 保存电力线点云: {output_dir}/powerlines_points.npy")
        
        if len(tower_points) > 0:
            np.save(f"{output_dir}/tower_points.npy", tower_points)
            print(f"✅ 保存塔架点云: {output_dir}/tower_points.npy")
        
        # 2. 保存为PLY格式
        try:
            import open3d as o3d
            
            if len(pl_points) > 0:
                pl_pcd = o3d.geometry.PointCloud()
                pl_pcd.points = o3d.utility.Vector3dVector(pl_points)
                pl_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
                o3d.io.write_point_cloud(f"{output_dir}/powerlines.ply", pl_pcd)
                print(f"✅ 保存电力线PLY: {output_dir}/powerlines.ply")
            
            if len(tower_points) > 0:
                tower_pcd = o3d.geometry.PointCloud()
                tower_pcd.points = o3d.utility.Vector3dVector(tower_points)
                tower_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
                o3d.io.write_point_cloud(f"{output_dir}/towers.ply", tower_pcd)
                print(f"✅ 保存塔架PLY: {output_dir}/towers.ply")
        
        except ImportError:
            print("⚠️  Open3D不可用，跳过PLY保存")
        
        # 3. 保存为文本格式
        if len(pl_points) > 0:
            np.savetxt(f"{output_dir}/powerlines_points.txt", pl_points, 
                      fmt='%.6f', header='X Y Z', comments='# ')
            print(f"✅ 保存电力线文本: {output_dir}/powerlines_points.txt")
        
        # 4. 保存电力线详细信息
        if power_lines:
            with open(f"{output_dir}/powerlines_info.txt", 'w') as f:
                f.write("电力线检测结果详情\\n")
                f.write("=" * 40 + "\\n\\n")
                
                for i, pl in enumerate(power_lines):
                    f.write(f"电力线 {i}:\\n")
                    f.write(f"  ID: {pl.get('powerline_id', i)}\\n")
                    f.write(f"  长度: {pl.get('total_length', 0):.2f}m\\n")
                    f.write(f"  点数: {len(pl.get('point_indices', []))}\\n")
                    f.write(f"  平均高度: {pl.get('height_stats', {}).get('mean', 0):.2f}m\\n")
                    f.write(f"  高度范围: {pl.get('height_stats', {}).get('min', 0):.2f} - {pl.get('height_stats', {}).get('max', 0):.2f}m\\n")
                    f.write("\\n")
            
            print(f"✅ 保存电力线信息: {output_dir}/powerlines_info.txt")
        
        # 5. 尝试保存为简单的LAS格式
        if len(pl_points) > 0:
            try:
                import laspy
                las_out = laspy.create()
                las_out.x = pl_points[:, 0]
                las_out.y = pl_points[:, 1]
                las_out.z = pl_points[:, 2]
                las_out.write(f"{output_dir}/powerlines_simple.las")
                print(f"✅ 保存电力线LAS: {output_dir}/powerlines_simple.las")
            except Exception as e:
                print(f"⚠️  LAS保存失败: {e}")
        
        print(f"\\n🎉 结果保存完成！输出目录: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = save_detection_results()