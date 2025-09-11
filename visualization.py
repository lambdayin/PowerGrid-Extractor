#!/usr/bin/env python3
"""
可视化模块 - 用于调试PowerGrid-Extractor算法的各个步骤
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import Dict, List, Tuple, Optional
import os

class PowerGridVisualizer:
    """电力网格提取算法可视化工具"""
    
    def __init__(self, save_dir: str = "debug_visualizations"):
        """初始化可视化工具
        
        Args:
            save_dir: 保存可视化结果的目录
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def visualize_original_pointcloud(self, points: np.ndarray, title: str = "Original Point Cloud", 
                                    save_name: str = "01_original_pointcloud.png"):
        """可视化原始点云
        
        Args:
            points: 点云数据 (N, 3)
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化原始点云: {len(points):,} 点")
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D视图
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(points[::100, 0], points[::100, 1], points[::100, 2], 
                            c=points[::100, 2], cmap='viridis', s=1)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D View')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
        
        # XY平面视图
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(points[::100, 0], points[::100, 1], 
                             c=points[::100, 2], cmap='viridis', s=1)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View (XY)')
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2)
        
        # 高度分布
        ax3 = fig.add_subplot(133)
        ax3.hist(points[:, 2], bins=50, alpha=0.7, color='blue')
        ax3.set_xlabel('Height (m)')
        ax3.set_ylabel('Count')
        ax3.set_title('Height Distribution')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  高度范围: {points[:, 2].min():.2f} - {points[:, 2].max():.2f} m")
        print(f"  XY范围: X({points[:, 0].min():.2f}, {points[:, 0].max():.2f}), Y({points[:, 1].min():.2f}, {points[:, 1].max():.2f})")
    
    def visualize_preprocessed_points(self, original_points: np.ndarray, filtered_points: np.ndarray,
                                    delta_h_min: float, title: str = "Preprocessing Results",
                                    save_name: str = "02_preprocessing_results.png"):
        """可视化预处理结果
        
        Args:
            original_points: 原始点云
            filtered_points: 过滤后点云
            delta_h_min: 最小高度阈值
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化预处理结果: {len(original_points):,} -> {len(filtered_points):,} 点")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 原始点云 XY视图
        ax1 = fig.add_subplot(221)
        ax1.scatter(original_points[::100, 0], original_points[::100, 1], 
                   c=original_points[::100, 2], cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Original Points ({len(original_points):,})')
        ax1.set_aspect('equal')
        
        # 过滤后点云 XY视图
        ax2 = fig.add_subplot(222)
        scatter = ax2.scatter(filtered_points[::50, 0], filtered_points[::50, 1], 
                            c=filtered_points[::50, 2], cmap='viridis', s=1)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Filtered Points ({len(filtered_points):,})')
        ax2.set_aspect('equal')
        plt.colorbar(scatter, ax=ax2)
        
        # 高度分布对比
        ax3 = fig.add_subplot(223)
        ax3.hist(original_points[:, 2], bins=50, alpha=0.5, label='Original', color='blue')
        ax3.hist(filtered_points[:, 2], bins=50, alpha=0.7, label='Filtered', color='red')
        ax3.axvline(x=delta_h_min, color='green', linestyle='--', linewidth=2, 
                   label=f'δh_min = {delta_h_min:.2f}m')
        ax3.set_xlabel('Height (m)')
        ax3.set_ylabel('Count')
        ax3.set_title('Height Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 过滤统计
        ax4 = fig.add_subplot(224)
        stats = {
            'Original': len(original_points),
            'Filtered': len(filtered_points),
            'Removed': len(original_points) - len(filtered_points)
        }
        bars = ax4.bar(stats.keys(), stats.values(), color=['blue', 'green', 'red'], alpha=0.7)
        ax4.set_ylabel('Point Count')
        ax4.set_title('Filtering Statistics')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        removal_rate = (len(original_points) - len(filtered_points)) / len(original_points) * 100
        print(f"  过滤掉 {removal_rate:.1f}% 的点")
        print(f"  δh_min = {delta_h_min:.2f} m")

    def visualize_voxel_grid(self, voxel_hash_3d: Dict, grid_2d: Dict, points: np.ndarray,
                           config, title: str = "Voxel Grid Structure",
                           save_name: str = "03_voxel_grid.png",
                           grid_origin: Optional[Tuple[float, float]] = None,
                           voxel_origin: Optional[Tuple[float, float, float]] = None):
        """可视化体素网格结构
        
        Args:
            voxel_hash_3d: 3D体素哈希表
            grid_2d: 2D网格
            points: 点云数据
            config: 配置对象
            title: 图标题
            save_name: 保存文件名
            grid_origin: 2D网格原点 (min_x, min_y)，与预处理一致（可选）
            voxel_origin: 3D体素原点 (min_x, min_y, min_z)，与预处理一致（可选）
        """
        print(f"🎨 可视化体素网格: {len(voxel_hash_3d)} 个3D体素, {len(grid_2d)} 个2D网格")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 计算/回退原点（与预处理保持一致）
        if points is not None and len(points) > 0:
            pts_min = points.min(axis=0)
        else:
            pts_min = np.array([0.0, 0.0, 0.0])
        
        if grid_origin is None:
            origin_x_2d = float(pts_min[0])
            origin_y_2d = float(pts_min[1])
        else:
            origin_x_2d = float(grid_origin[0])
            origin_y_2d = float(grid_origin[1])
        
        if voxel_origin is None:
            origin_x_3d = float(pts_min[0])
            origin_y_3d = float(pts_min[1])
            origin_z_3d = float(pts_min[2])
        else:
            origin_x_3d = float(voxel_origin[0])
            origin_y_3d = float(voxel_origin[1])
            origin_z_3d = float(voxel_origin[2])
        
        # 2D网格可视化
        ax1 = fig.add_subplot(221)
        grid_centers = []
        grid_point_counts = []
        
        for grid_key, point_indices in grid_2d.items():
            x_idx, y_idx = grid_key
            center_x = origin_x_2d + x_idx * config.grid_2d_size + config.grid_2d_size/2
            center_y = origin_y_2d + y_idx * config.grid_2d_size + config.grid_2d_size/2
            grid_centers.append([center_x, center_y])
            grid_point_counts.append(len(point_indices))
        
        if grid_centers:
            grid_centers = np.array(grid_centers)
            scatter = ax1.scatter(grid_centers[:, 0], grid_centers[:, 1], 
                                c=grid_point_counts, cmap='viridis', s=50)
            plt.colorbar(scatter, ax=ax1, label='Points per grid')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title(f'2D Grid ({len(grid_2d)} cells)')
            ax1.set_aspect('equal')
        
        # 3D体素中心可视化
        ax2 = fig.add_subplot(222, projection='3d')
        voxel_centers = []
        voxel_point_counts = []
        
        for voxel_key, point_indices in voxel_hash_3d.items():
            x_idx, y_idx, z_idx = voxel_key
            center_x = origin_x_3d + x_idx * config.voxel_size + config.voxel_size/2
            center_y = origin_y_3d + y_idx * config.voxel_size + config.voxel_size/2
            center_z = origin_z_3d + z_idx * config.voxel_size + config.voxel_size/2
            voxel_centers.append([center_x, center_y, center_z])
            voxel_point_counts.append(len(point_indices))
        
        if voxel_centers:
            voxel_centers = np.array(voxel_centers)
            scatter = ax2.scatter(voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2],
                                c=voxel_point_counts, cmap='viridis', s=20)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_zlabel('Z (m)')
            ax2.set_title(f'3D Voxels ({len(voxel_hash_3d)} voxels)')
        
        # 体素大小分布
        ax3 = fig.add_subplot(223)
        if voxel_point_counts:
            ax3.hist(voxel_point_counts, bins=30, alpha=0.7, color='blue')
            ax3.set_xlabel('Points per Voxel')
            ax3.set_ylabel('Voxel Count')
            ax3.set_title('Voxel Size Distribution')
            ax3.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_size = np.mean(voxel_point_counts)
            ax3.axvline(x=mean_size, color='red', linestyle='--', 
                       label=f'Mean: {mean_size:.1f}')
            ax3.legend()
        
        # 高度分布（按体素）
        ax4 = fig.add_subplot(224)
        if voxel_centers is not None and len(voxel_centers) > 0:
            ax4.hist(voxel_centers[:, 2], bins=30, alpha=0.7, color='green')
            ax4.set_xlabel('Voxel Height (m)')
            ax4.set_ylabel('Voxel Count')
            ax4.set_title('Voxel Height Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        if voxel_point_counts:
            print(f"  平均每个体素: {np.mean(voxel_point_counts):.1f} 点")
            print(f"  体素大小: {config.voxel_size} m")
            print(f"  2D网格大小: {config.grid_2d_size} m")

    def visualize_linear_voxels(self, linear_voxels: Dict, all_voxel_features: Dict, 
                              voxel_hash_3d: Dict, config, delta_h_min: float,
                              title: str = "Linear Voxel Analysis",
                              save_name: str = "04_linear_voxels.png"):
        """可视化线性体素分析结果
        
        Args:
            linear_voxels: 线性体素字典
            all_voxel_features: 所有体素特征
            voxel_hash_3d: 3D体素哈希表
            config: 配置对象
            delta_h_min: 最小高度阈值
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化线性体素: {len(linear_voxels)} 个线性体素 / {len(all_voxel_features)} 总体素")
        
        fig = plt.figure(figsize=(15, 12))
        
        # 所有体素的a1D分布
        ax1 = fig.add_subplot(231)
        a1d_values = [f['a1D'] for f in all_voxel_features.values()]
        ax1.hist(a1d_values, bins=50, alpha=0.7, color='blue', label='All voxels')
        ax1.axvline(x=config.a1d_linear_thr, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold = {config.a1d_linear_thr}')
        ax1.set_xlabel('a1D (Linearity)')
        ax1.set_ylabel('Voxel Count')
        ax1.set_title('a1D Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 线性体素的3D分布
        ax2 = fig.add_subplot(232, projection='3d')
        if linear_voxels:
            linear_centers = []
            linear_a1d = []
            
            for voxel_idx, features in linear_voxels.items():
                if 'centroid' in features:
                    linear_centers.append(features['centroid'])
                    linear_a1d.append(features['a1D'])
            
            if linear_centers:
                linear_centers = np.array(linear_centers)
                scatter = ax2.scatter(linear_centers[:, 0], linear_centers[:, 1], linear_centers[:, 2],
                                    c=linear_a1d, cmap='hot', s=30)
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_zlabel('Z (m)')
                ax2.set_title('Linear Voxels 3D')
                plt.colorbar(scatter, ax=ax2, shrink=0.5, label='a1D')
        
        # 线性体素XY视图
        ax3 = fig.add_subplot(233)
        if linear_voxels:
            if linear_centers is not None and len(linear_centers) > 0:
                scatter = ax3.scatter(linear_centers[:, 0], linear_centers[:, 1],
                                    c=linear_centers[:, 2], cmap='viridis', s=50)
                ax3.set_xlabel('X (m)')
                ax3.set_ylabel('Y (m)')
                ax3.set_title('Linear Voxels Top View')
                ax3.set_aspect('equal')
                plt.colorbar(scatter, ax=ax3, label='Height (m)')
        
        # 线性体素高度分布
        ax4 = fig.add_subplot(234)
        if linear_voxels:
            linear_heights = [f['centroid'][2] for f in linear_voxels.values() if 'centroid' in f]
            if linear_heights:
                ax4.hist(linear_heights, bins=20, alpha=0.7, color='green')
                ax4.axvline(x=delta_h_min, color='red', linestyle='--', 
                           linewidth=2, label=f'δh_min = {delta_h_min:.2f}m')
                ax4.set_xlabel('Height (m)')
                ax4.set_ylabel('Linear Voxel Count')
                ax4.set_title('Linear Voxel Height Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 线性度阈值分析
        ax5 = fig.add_subplot(235)
        thresholds = np.arange(0.1, 1.0, 0.05)
        counts = []
        for thr in thresholds:
            count = sum(1 for a1d in a1d_values if a1d > thr)
            counts.append(count)
        
        ax5.plot(thresholds, counts, 'b-o', markersize=4)
        ax5.axvline(x=config.a1d_linear_thr, color='red', linestyle='--', 
                   linewidth=2, label=f'Current = {config.a1d_linear_thr}')
        ax5.set_xlabel('a1D Threshold')
        ax5.set_ylabel('Linear Voxel Count')
        ax5.set_title('Threshold Sensitivity Analysis')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 统计表
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        stats_text = f"""Linear Voxel Statistics:
        
Total Voxels: {len(all_voxel_features):,}
Linear Voxels: {len(linear_voxels):,}
Linear Ratio: {len(linear_voxels)/len(all_voxel_features)*100:.1f}%

a1D Threshold: {config.a1d_linear_thr}
Mean a1D: {np.mean(a1d_values):.3f}
Max a1D: {np.max(a1d_values):.3f}

Height Filter (δh_min): {delta_h_min:.2f}m"""

        if linear_voxels:
            linear_heights = [f['centroid'][2] for f in linear_voxels.values() if 'centroid' in f]
            if linear_heights:
                above_threshold = sum(1 for h in linear_heights if h > delta_h_min)
                stats_text += f"""
Linear Voxels > δh_min: {above_threshold}
Height Range: {min(linear_heights):.1f} - {max(linear_heights):.1f}m"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        if linear_voxels and linear_heights:
            print(f"  线性体素高度范围: {min(linear_heights):.2f} - {max(linear_heights):.2f} m")
            above_threshold = sum(1 for h in linear_heights if h > delta_h_min)
            print(f"  高于阈值的线性体素: {above_threshold}/{len(linear_heights)}")

    def visualize_power_line_segments(self, segments: List[Dict], points: np.ndarray,
                                    title: str = "Power Line Segments",
                                    save_name: str = "05_power_line_segments.png"):
        """可视化电力线段
        
        Args:
            segments: 电力线段列表
            points: 原始点云
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化电力线段: {len(segments)} 个段")
        
        if not segments:
            print("  ❌ 没有段可以可视化")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D视图 - 所有段
        ax1 = fig.add_subplot(221, projection='3d')
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        
        # 显示背景点云（采样）
        bg_sample = points[::1000] if len(points) > 10000 else points
        ax1.scatter(bg_sample[:, 0], bg_sample[:, 1], bg_sample[:, 2], 
                   c='lightgray', s=1, alpha=0.3)
        
        for i, segment in enumerate(segments):
            if 'point_indices' in segment and len(segment['point_indices']) > 0:
                seg_points = points[segment['point_indices']]
                ax1.scatter(seg_points[:, 0], seg_points[:, 1], seg_points[:, 2],
                           c=[colors[i]], s=20, label=f'Segment {i}')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D View - All Segments')
        if len(segments) <= 10:
            ax1.legend()
        
        # XY平面视图
        ax2 = fig.add_subplot(222)
        ax2.scatter(bg_sample[:, 0], bg_sample[:, 1], c='lightgray', s=1, alpha=0.3)
        
        for i, segment in enumerate(segments):
            if 'point_indices' in segment and len(segment['point_indices']) > 0:
                seg_points = points[segment['point_indices']]
                ax2.scatter(seg_points[:, 0], seg_points[:, 1],
                           c=[colors[i]], s=20, label=f'Segment {i}')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View - All Segments')
        ax2.set_aspect('equal')
        
        # 段长度分布
        ax3 = fig.add_subplot(223)
        lengths = [seg.get('length', 0) for seg in segments]
        if lengths:
            ax3.hist(lengths, bins=min(20, len(segments)), alpha=0.7, color='blue')
            ax3.set_xlabel('Segment Length (m)')
            ax3.set_ylabel('Count')
            ax3.set_title('Segment Length Distribution')
            ax3.grid(True, alpha=0.3)
            
            # 添加统计线
            mean_length = np.mean(lengths)
            ax3.axvline(x=mean_length, color='red', linestyle='--', 
                       label=f'Mean: {mean_length:.1f}m')
            ax3.legend()
        
        # 段统计信息
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        if segments:
            lengths = [seg.get('length', 0) for seg in segments]
            point_counts = [len(seg.get('point_indices', [])) for seg in segments]
            
            stats_text = f"""Segment Statistics:
            
Total Segments: {len(segments)}
            
Length Stats:
  Mean: {np.mean(lengths):.2f} m
  Min: {min(lengths):.2f} m
  Max: {max(lengths):.2f} m
  Total: {sum(lengths):.2f} m

Points per Segment:
  Mean: {np.mean(point_counts):.1f}
  Min: {min(point_counts)}
  Max: {max(point_counts)}
  Total: {sum(point_counts):,}"""
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        if segments:
            lengths = [seg.get('length', 0) for seg in segments]
            print(f"  段长度范围: {min(lengths):.2f} - {max(lengths):.2f} m")
            print(f"  平均段长度: {np.mean(lengths):.2f} m")

    def visualize_final_results(self, power_lines: List[Dict], filtered_lines: List[Dict], 
                              points: np.ndarray, title: str = "Final Power Line Results",
                              save_name: str = "06_final_results.png"):
        """可视化最终电力线提取结果
        
        Args:
            power_lines: 合并前的电力线
            filtered_lines: 过滤后的最终电力线
            points: 原始点云
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化最终结果: {len(power_lines)} -> {len(filtered_lines)} 条电力线")
        
        fig = plt.figure(figsize=(15, 12))
        
        # 背景点云采样
        bg_sample = points[::1000] if len(points) > 10000 else points
        
        # 合并前的电力线 - 3D视图
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(bg_sample[:, 0], bg_sample[:, 1], bg_sample[:, 2], 
                   c='lightgray', s=1, alpha=0.2)
        
        if power_lines:
            colors = plt.cm.tab10(np.linspace(0, 1, len(power_lines)))
            for i, pl in enumerate(power_lines):
                if 'point_indices' in pl and len(pl['point_indices']) > 0:
                    pl_points = points[pl['point_indices']]
                    ax1.scatter(pl_points[:, 0], pl_points[:, 1], pl_points[:, 2],
                               c=[colors[i]], s=15, label=f'Line {i}')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Before Filtering ({len(power_lines)} lines)')
        
        # 过滤后的电力线 - 3D视图
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.scatter(bg_sample[:, 0], bg_sample[:, 1], bg_sample[:, 2], 
                   c='lightgray', s=1, alpha=0.2)
        
        if filtered_lines:
            colors = plt.cm.Set1(np.linspace(0, 1, len(filtered_lines)))
            for i, pl in enumerate(filtered_lines):
                if 'point_indices' in pl and len(pl['point_indices']) > 0:
                    pl_points = points[pl['point_indices']]
                    ax2.scatter(pl_points[:, 0], pl_points[:, 1], pl_points[:, 2],
                               c=[colors[i]], s=20, label=f'Final Line {i}')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title(f'After Filtering ({len(filtered_lines)} lines)')
        if len(filtered_lines) <= 5:
            ax2.legend()
        
        # XY平面视图 - 最终结果
        ax3 = fig.add_subplot(233)
        ax3.scatter(bg_sample[:, 0], bg_sample[:, 1], c='lightgray', s=1, alpha=0.3)
        
        if filtered_lines:
            for i, pl in enumerate(filtered_lines):
                if 'point_indices' in pl and len(pl['point_indices']) > 0:
                    pl_points = points[pl['point_indices']]
                    ax3.scatter(pl_points[:, 0], pl_points[:, 1],
                               c=[colors[i]], s=20, label=f'Line {i}')
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Final Results - Top View')
        ax3.set_aspect('equal')
        
        # 电力线长度对比
        ax4 = fig.add_subplot(234)
        if power_lines or filtered_lines:
            original_lengths = [pl.get('total_length', 0) for pl in power_lines] if power_lines else []
            final_lengths = [pl.get('total_length', 0) for pl in filtered_lines] if filtered_lines else []
            
            x_pos = np.arange(max(len(original_lengths), len(final_lengths)))
            width = 0.35
            
            if original_lengths:
                ax4.bar(x_pos - width/2, original_lengths + [0] * (len(x_pos) - len(original_lengths)), 
                       width, label='Before Filtering', alpha=0.7, color='blue')
            if final_lengths:
                ax4.bar(x_pos + width/2, final_lengths + [0] * (len(x_pos) - len(final_lengths)), 
                       width, label='After Filtering', alpha=0.7, color='red')
            
            ax4.set_xlabel('Power Line Index')
            ax4.set_ylabel('Length (m)')
            ax4.set_title('Power Line Lengths Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 高度分布 - 最终电力线
        ax5 = fig.add_subplot(235)
        if filtered_lines:
            all_heights = []
            for pl in filtered_lines:
                if 'point_indices' in pl and len(pl['point_indices']) > 0:
                    pl_points = points[pl['point_indices']]
                    all_heights.extend(pl_points[:, 2])
            
            if all_heights:
                ax5.hist(all_heights, bins=30, alpha=0.7, color='green')
                ax5.set_xlabel('Height (m)')
                ax5.set_ylabel('Point Count')
                ax5.set_title('Final Power Lines Height Distribution')
                ax5.grid(True, alpha=0.3)
        
        # 统计摘要
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        stats_text = f"""Final Results Summary:
        
Original Lines: {len(power_lines) if power_lines else 0}
Final Lines: {len(filtered_lines) if filtered_lines else 0}
Filtering Rate: {(1 - len(filtered_lines)/max(1, len(power_lines)))*100:.1f}%"""

        if filtered_lines:
            total_length = sum(pl.get('total_length', 0) for pl in filtered_lines)
            total_points = sum(len(pl.get('point_indices', [])) for pl in filtered_lines)
            avg_length = total_length / len(filtered_lines) if filtered_lines else 0
            
            stats_text += f"""

Final Statistics:
  Total Length: {total_length:.1f} m
  Average Length: {avg_length:.1f} m
  Total Points: {total_points:,}
  Avg Points/Line: {total_points/len(filtered_lines):.0f}"""
            
            if filtered_lines:
                lengths = [pl.get('total_length', 0) for pl in filtered_lines]
                stats_text += f"""
  
Length Range:
  Min: {min(lengths):.1f} m
  Max: {max(lengths):.1f} m"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        if filtered_lines:
            total_length = sum(pl.get('total_length', 0) for pl in filtered_lines)
            print(f"  最终提取到 {len(filtered_lines)} 条电力线")
            print(f"  总长度: {total_length:.1f} m")
            print(f"  平均长度: {total_length/len(filtered_lines):.1f} m")
        else:
            print("  ❌ 没有最终的电力线结果")

    def create_summary_report(self, stats: Dict, save_name: str = "00_summary_report.png"):
        """创建算法执行的总结报告
        
        Args:
            stats: 包含各步骤统计信息的字典
            save_name: 保存文件名
        """
        print("🎨 创建总结报告")
        
        fig = plt.figure(figsize=(15, 8))
        
        # 算法流程图（文本形式）
        ax1 = fig.add_subplot(121)
        ax1.axis('off')
        
        pipeline_text = f"""PowerGrid Extraction Pipeline Summary

📊 INPUT
  • Original Points: {stats.get('original_points', 'N/A'):,}
  • Height Range: {stats.get('height_range', 'N/A')}
  
🔧 PREPROCESSING  
  • Filtered Points: {stats.get('filtered_points', 'N/A'):,}
  • Removal Rate: {stats.get('removal_rate', 'N/A'):.1f}%
  • δh_min: {stats.get('delta_h_min', 'N/A'):.2f} m
  
🔲 VOXELIZATION
  • 2D Grids: {stats.get('grid_2d_count', 'N/A'):,}
  • 3D Voxels: {stats.get('voxel_3d_count', 'N/A'):,}
  • Voxel Size: {stats.get('voxel_size', 'N/A')} m
  
📐 FEATURE ANALYSIS
  • Linear Voxels: {stats.get('linear_voxels', 'N/A'):,}
  • Linearity Threshold: {stats.get('a1d_threshold', 'N/A')}
  • Linear Ratio: {stats.get('linear_ratio', 'N/A'):.1f}%
  
🔗 SEGMENTATION
  • Local Segments: {stats.get('segments', 'N/A'):,}
  • Graph Nodes: {stats.get('graph_nodes', 'N/A'):,}
  • Graph Edges: {stats.get('graph_edges', 'N/A'):,}
  
⚡ FINAL RESULTS
  • Power Lines (raw): {stats.get('raw_powerlines', 'N/A'):,}
  • Power Lines (filtered): {stats.get('final_powerlines', 'N/A'):,}
  • Total Length: {stats.get('total_length', 'N/A'):.1f} m"""
        
        ax1.text(0.05, 0.95, pipeline_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 处理效率分析
        ax2 = fig.add_subplot(122)
        
        # 创建数据保留率的瀑布图
        stages = ['Original\nPoints', 'After\nFiltering', 'In Linear\nVoxels', 'In Segments', 'Final\nLines']
        values = [
            stats.get('original_points', 0),
            stats.get('filtered_points', 0), 
            stats.get('linear_points', 0),
            stats.get('segment_points', 0),
            stats.get('final_points', 0)
        ]
        
        # 确保所有值都有效
        valid_values = [v for v in values if v > 0]
        if len(valid_values) >= 2:
            bars = ax2.bar(range(len(values)), values, color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{val:,}', ha='center', va='bottom', fontsize=9)
            
            ax2.set_xticks(range(len(stages)))
            ax2.set_xticklabels(stages, rotation=45, ha='right')
            ax2.set_ylabel('Point Count')
            ax2.set_title('Data Processing Pipeline')
            ax2.grid(True, alpha=0.3)
            
            # 添加连接线显示数据流
            for i in range(len(values)-1):
                if values[i] > 0 and values[i+1] > 0:
                    ax2.annotate('', xy=(i+1, values[i+1]), xytext=(i, values[i]),
                               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        plt.suptitle('PowerGrid Extraction Algorithm Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  报告已保存到: {os.path.join(self.save_dir, save_name)}")