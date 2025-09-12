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
  
🔗 POWER LINES
  • Local Segments: {stats.get('segments', 'N/A'):,}
  • Graph Nodes: {stats.get('graph_nodes', 'N/A'):,}
  • Graph Edges: {stats.get('graph_edges', 'N/A'):,}
  • Final Lines: {stats.get('final_powerlines', 'N/A'):,}
  • Total Length: {stats.get('total_length', 'N/A'):.1f} m
  
🗼 TOWER DETECTION
  • Step1 Candidates: {stats.get('tower_step1_candidates', 'N/A'):,}
  • Step2 Candidates: {stats.get('tower_step2_candidates', 'N/A'):,}
  • Step3 Candidates: {stats.get('tower_step3_candidates', 'N/A'):,}
  • Tower Clusters: {stats.get('tower_clusters', 'N/A'):,}
  • Final Towers: {stats.get('final_towers', 'N/A'):,}
  • Tower Points: {stats.get('tower_points', 'N/A'):,}"""
        
        ax1.text(0.05, 0.95, pipeline_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 处理效率分析
        ax2 = fig.add_subplot(122)
        
        # 创建数据保留率的瀑布图
        stages = ['Original\nPoints', 'After\nFiltering', 'In Linear\nVoxels', 'In Segments', 'Final\nLines', 'Tower\nPoints']
        values = [
            stats.get('original_points', 0),
            stats.get('filtered_points', 0), 
            stats.get('linear_points', 0),
            stats.get('segment_points', 0),
            stats.get('final_points', 0),
            stats.get('tower_points', 0)
        ]
        
        # 确保所有值都有效
        valid_values = [v for v in values if v > 0]
        if len(valid_values) >= 2:
            bars = ax2.bar(range(len(values)), values, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'], alpha=0.7)
            
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

    def visualize_tower_step1_initial_screening(self, candidate_grids: set, grid_features: Dict, 
                                              delta_h_min: float, tower_head_height: float,
                                              title: str = "Tower Step 1: Initial Height Screening",
                                              save_name: str = "07_tower_step1_screening.png"):
        """可视化塔检测步骤1：初始高度差筛选
        
        Args:
            candidate_grids: 候选网格集合
            grid_features: 网格特征
            delta_h_min: 最小高度间隙
            tower_head_height: 塔头高度
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化塔检测步骤1: {len(candidate_grids)} 个候选网格")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 所有网格的高度差分布
        ax1 = fig.add_subplot(221)
        all_height_diffs = [features.get('HeightDiff', 0) for features in grid_features.values()]
        threshold = max(delta_h_min, tower_head_height * 0.5)
        
        ax1.hist(all_height_diffs, bins=50, alpha=0.7, color='lightblue', label='All grids')
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold:.1f}m')
        ax1.set_xlabel('Height Difference (m)')
        ax1.set_ylabel('Grid Count')
        ax1.set_title('Grid Height Difference Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 候选网格的空间分布
        ax2 = fig.add_subplot(222)
        if candidate_grids:
            candidate_coords = []
            candidate_heights = []
            for grid_idx in candidate_grids:
                if grid_idx in grid_features:
                    features = grid_features[grid_idx]
                    if 'centroid' in features:
                        candidate_coords.append(features['centroid'][:2])
                        candidate_heights.append(features.get('HeightDiff', 0))
            
            if candidate_coords:
                candidate_coords = np.array(candidate_coords)
                scatter = ax2.scatter(candidate_coords[:, 0], candidate_coords[:, 1], 
                                    c=candidate_heights, cmap='hot', s=50)
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_title('Candidate Grid Locations')
                ax2.set_aspect('equal')
                plt.colorbar(scatter, ax=ax2, label='Height Diff (m)')
        
        # 候选网格的高度差分布
        ax3 = fig.add_subplot(223)
        if candidate_grids:
            candidate_height_diffs = []
            for grid_idx in candidate_grids:
                if grid_idx in grid_features:
                    candidate_height_diffs.append(grid_features[grid_idx].get('HeightDiff', 0))
            
            if candidate_height_diffs:
                ax3.hist(candidate_height_diffs, bins=min(20, len(candidate_height_diffs)), 
                        alpha=0.7, color='orange')
                ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
                ax3.set_xlabel('Height Difference (m)')
                ax3.set_ylabel('Candidate Grid Count')
                ax3.set_title('Candidate Grid Height Distribution')
                ax3.grid(True, alpha=0.3)
        
        # 统计信息
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats_text = f"""Step 1 Statistics:
        
Total Grids: {len(grid_features):,}
Candidate Grids: {len(candidate_grids):,}
Selection Rate: {len(candidate_grids)/len(grid_features)*100:.1f}%

Parameters:
  δh_min: {delta_h_min:.2f} m
  Tower Head Height: {tower_head_height:.2f} m
  Threshold: {threshold:.2f} m"""
        
        if all_height_diffs:
            stats_text += f"""

Height Diff Stats:
  Mean: {np.mean(all_height_diffs):.2f} m
  Max: {np.max(all_height_diffs):.2f} m
  >Threshold: {sum(1 for h in all_height_diffs if h > threshold)} grids"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_tower_step2_window_refinement(self, initial_candidates: set, refined_candidates: set,
                                              grid_features: Dict, tower_head_height: float,
                                              title: str = "Tower Step 2: Moving Window Refinement",
                                              save_name: str = "08_tower_step2_refinement.png"):
        """可视化塔检测步骤2：移动窗口细化
        
        Args:
            initial_candidates: 初始候选网格
            refined_candidates: 细化后的候选网格
            grid_features: 网格特征
            tower_head_height: 塔头高度
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化塔检测步骤2: {len(initial_candidates)} -> {len(refined_candidates)} 个候选网格")
        
        fig = plt.figure(figsize=(15, 8))
        
        # 细化前后对比 - 空间分布
        ax1 = fig.add_subplot(131)
        
        # 初始候选
        if initial_candidates:
            initial_coords = []
            for grid_idx in initial_candidates:
                if grid_idx in grid_features and 'centroid' in grid_features[grid_idx]:
                    initial_coords.append(grid_features[grid_idx]['centroid'][:2])
            
            if initial_coords:
                initial_coords = np.array(initial_coords)
                ax1.scatter(initial_coords[:, 0], initial_coords[:, 1], 
                           c='lightblue', s=30, alpha=0.7, label=f'Initial ({len(initial_candidates)})')
        
        # 细化后候选
        if refined_candidates:
            refined_coords = []
            for grid_idx in refined_candidates:
                if grid_idx in grid_features and 'centroid' in grid_features[grid_idx]:
                    refined_coords.append(grid_features[grid_idx]['centroid'][:2])
            
            if refined_coords:
                refined_coords = np.array(refined_coords)
                ax1.scatter(refined_coords[:, 0], refined_coords[:, 1], 
                           c='red', s=50, alpha=0.8, label=f'Refined ({len(refined_candidates)})')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Spatial Distribution Comparison')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 被移除的候选网格
        ax2 = fig.add_subplot(132)
        removed_candidates = initial_candidates - refined_candidates
        
        if removed_candidates:
            removed_coords = []
            removed_heights = []
            for grid_idx in removed_candidates:
                if grid_idx in grid_features:
                    features = grid_features[grid_idx]
                    if 'centroid' in features:
                        removed_coords.append(features['centroid'][:2])
                        removed_heights.append(features.get('HeightDiff', 0))
            
            if removed_coords:
                removed_coords = np.array(removed_coords)
                scatter = ax2.scatter(removed_coords[:, 0], removed_coords[:, 1], 
                                    c=removed_heights, cmap='Reds', s=30, alpha=0.7)
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_title(f'Removed Candidates ({len(removed_candidates)})')
                ax2.set_aspect('equal')
                plt.colorbar(scatter, ax=ax2, label='Height Diff (m)')
        
        # 统计信息和参数
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        retention_rate = len(refined_candidates) / max(1, len(initial_candidates)) * 100
        removal_rate = 100 - retention_rate
        
        stats_text = f"""Step 2 Statistics:
        
Initial Candidates: {len(initial_candidates):,}
Refined Candidates: {len(refined_candidates):,}
Removed: {len(removed_candidates):,}

Retention Rate: {retention_rate:.1f}%
Removal Rate: {removal_rate:.1f}%

Parameters:
  Window Size: 2×2
  Tower Head Height: {tower_head_height:.2f} m
  Height Variance Check: Enabled"""
        
        # 添加高度差统计
        if refined_candidates:
            refined_heights = []
            for grid_idx in refined_candidates:
                if grid_idx in grid_features:
                    refined_heights.append(grid_features[grid_idx].get('HeightDiff', 0))
            
            if refined_heights:
                stats_text += f"""
                
Refined Grid Heights:
  Mean: {np.mean(refined_heights):.2f} m
  Min: {min(refined_heights):.2f} m
  Max: {max(refined_heights):.2f} m"""
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_tower_step3_vertical_continuity(self, step2_candidates: set, step3_candidates: set,
                                                 grid_features: Dict, points: np.ndarray, 
                                                 tower_head_height: float,
                                                 title: str = "Tower Step 3: Vertical Continuity Check",
                                                 save_name: str = "09_tower_step3_continuity.png"):
        """可视化塔检测步骤3：垂直连续性检查
        
        Args:
            step2_candidates: 步骤2的候选网格
            step3_candidates: 步骤3的候选网格 
            grid_features: 网格特征
            points: 点云数据
            tower_head_height: 塔头高度
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化塔检测步骤3: {len(step2_candidates)} -> {len(step3_candidates)} 个候选网格")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 垂直连续性检查结果对比
        ax1 = fig.add_subplot(221)
        failed_candidates = step2_candidates - step3_candidates
        
        # 通过和失败的候选网格分布
        if step3_candidates:
            passed_coords = []
            for grid_idx in step3_candidates:
                if grid_idx in grid_features and 'centroid' in grid_features[grid_idx]:
                    passed_coords.append(grid_features[grid_idx]['centroid'][:2])
            
            if passed_coords:
                passed_coords = np.array(passed_coords)
                ax1.scatter(passed_coords[:, 0], passed_coords[:, 1], 
                           c='green', s=50, alpha=0.8, label=f'Passed ({len(step3_candidates)})')
        
        if failed_candidates:
            failed_coords = []
            for grid_idx in failed_candidates:
                if grid_idx in grid_features and 'centroid' in grid_features[grid_idx]:
                    failed_coords.append(grid_features[grid_idx]['centroid'][:2])
            
            if failed_coords:
                failed_coords = np.array(failed_coords)
                ax1.scatter(failed_coords[:, 0], failed_coords[:, 1], 
                           c='red', s=30, alpha=0.6, label=f'Failed ({len(failed_candidates)})', marker='x')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Vertical Continuity Check Results')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 高度分析：通过vs失败的候选
        ax2 = fig.add_subplot(222)
        if step3_candidates and failed_candidates:
            passed_max_heights = []
            failed_max_heights = []
            
            for grid_idx in step3_candidates:
                if grid_idx in grid_features:
                    features = grid_features[grid_idx]
                    passed_max_heights.append(features.get('max_height', 0))
            
            for grid_idx in failed_candidates:
                if grid_idx in grid_features:
                    features = grid_features[grid_idx]
                    failed_max_heights.append(features.get('max_height', 0))
            
            if passed_max_heights:
                ax2.hist(passed_max_heights, bins=20, alpha=0.7, color='green', 
                        label=f'Passed ({len(passed_max_heights)})')
            if failed_max_heights:
                ax2.hist(failed_max_heights, bins=20, alpha=0.7, color='red', 
                        label=f'Failed ({len(failed_max_heights)})')
            
            ax2.axvline(x=tower_head_height, color='blue', linestyle='--', 
                       linewidth=2, label=f'Height Threshold = {tower_head_height:.1f}m')
            ax2.set_xlabel('Maximum Height (m)')
            ax2.set_ylabel('Grid Count')
            ax2.set_title('Height Distribution Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 垂直结构示例（选择几个代表性候选）
        ax3 = fig.add_subplot(223)
        if step3_candidates:
            sample_candidates = list(step3_candidates)[:3]  # 取前3个作为示例
            colors = ['green', 'blue', 'purple']
            
            for i, grid_idx in enumerate(sample_candidates):
                if grid_idx in grid_features:
                    # 这里需要获取该网格的点云数据来展示垂直结构
                    # 简化版本：显示网格的高度范围
                    features = grid_features[grid_idx]
                    if 'centroid' in features:
                        centroid = features['centroid']
                        height_diff = features.get('HeightDiff', 0)
                        
                        # 模拟垂直结构显示
                        ax3.bar(i, height_diff, color=colors[i], alpha=0.7, 
                               label=f'Grid {grid_idx}')
            
            ax3.set_xlabel('Sample Grid Index')
            ax3.set_ylabel('Height Difference (m)')
            ax3.set_title('Vertical Structure Examples')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 统计信息
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        pass_rate = len(step3_candidates) / max(1, len(step2_candidates)) * 100
        fail_rate = 100 - pass_rate
        
        stats_text = f"""Step 3 Statistics:
        
Input Candidates: {len(step2_candidates):,}
Passed Continuity: {len(step3_candidates):,}
Failed Continuity: {len(failed_candidates):,}

Pass Rate: {pass_rate:.1f}%
Fail Rate: {fail_rate:.1f}%

Parameters:
  Max Height Gap: {tower_head_height:.2f} m
  Min Height Threshold: {tower_head_height:.2f} m"""
        
        if step3_candidates:
            # 计算通过候选的统计
            passed_heights = []
            for grid_idx in step3_candidates:
                if grid_idx in grid_features:
                    passed_heights.append(grid_features[grid_idx].get('HeightDiff', 0))
            
            if passed_heights:
                stats_text += f"""
                
Passed Candidates:
  Mean Height Diff: {np.mean(passed_heights):.2f} m
  Max Height Diff: {max(passed_heights):.2f} m"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_tower_step4_clustering(self, valid_candidates: set, tower_clusters: List[Dict],
                                        grid_features: Dict, 
                                        title: str = "Tower Step 4: Clustering to Towers",
                                        save_name: str = "10_tower_step4_clustering.png"):
        """可视化塔检测步骤4：聚类成塔
        
        Args:
            valid_candidates: 有效候选网格
            tower_clusters: 塔聚类结果
            grid_features: 网格特征
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化塔检测步骤4: {len(valid_candidates)} 个候选网格 -> {len(tower_clusters)} 个塔聚类")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 聚类结果空间分布
        ax1 = fig.add_subplot(221)
        
        if tower_clusters:
            colors = plt.cm.Set3(np.linspace(0, 1, len(tower_clusters)))
            
            for i, cluster in enumerate(tower_clusters):
                cluster_coords = []
                for grid_idx in cluster['grid_cells']:
                    if grid_idx in grid_features and 'centroid' in grid_features[grid_idx]:
                        cluster_coords.append(grid_features[grid_idx]['centroid'][:2])
                
                if cluster_coords:
                    cluster_coords = np.array(cluster_coords)
                    ax1.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                               c=[colors[i]], s=100, alpha=0.8, 
                               label=f'Tower {i} ({len(cluster["grid_cells"])} cells)')
                    
                    # 绘制聚类中心
                    centroid = cluster['centroid'][:2]
                    ax1.scatter(centroid[0], centroid[1], c='red', s=200, marker='*', 
                               edgecolor='black', linewidth=2)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Tower Cluster Spatial Distribution')
        if len(tower_clusters) <= 10:
            ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 聚类大小分布
        ax2 = fig.add_subplot(222)
        if tower_clusters:
            cluster_sizes = [len(cluster['grid_cells']) for cluster in tower_clusters]
            ax2.hist(cluster_sizes, bins=min(10, max(cluster_sizes)), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Number of Grid Cells per Cluster')
            ax2.set_ylabel('Cluster Count')
            ax2.set_title('Cluster Size Distribution')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计线
            mean_size = np.mean(cluster_sizes)
            ax2.axvline(x=mean_size, color='red', linestyle='--', 
                       label=f'Mean: {mean_size:.1f}')
            ax2.legend()
        
        # 聚类质量分析
        ax3 = fig.add_subplot(223)
        if tower_clusters:
            cluster_heights = [cluster['max_height_diff'] for cluster in tower_clusters]
            cluster_densities = [cluster['avg_density'] for cluster in tower_clusters]
            
            scatter = ax3.scatter(cluster_sizes, cluster_heights, 
                                c=cluster_densities, cmap='viridis', s=100, alpha=0.7)
            ax3.set_xlabel('Cluster Size (grid cells)')
            ax3.set_ylabel('Max Height Difference (m)')
            ax3.set_title('Cluster Quality Analysis')
            plt.colorbar(scatter, ax=ax3, label='Average Density')
            ax3.grid(True, alpha=0.3)
        
        # 聚类统计信息
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats_text = f"""Step 4 Statistics:
        
Valid Candidates: {len(valid_candidates):,}
Tower Clusters: {len(tower_clusters):,}
Clustering Rate: {len(tower_clusters)/max(1, len(valid_candidates))*100:.1f}%"""
        
        if tower_clusters:
            cluster_sizes = [len(cluster['grid_cells']) for cluster in tower_clusters]
            total_points = sum(cluster['total_points'] for cluster in tower_clusters)
            
            stats_text += f"""
            
Cluster Statistics:
  Total Clustered Cells: {sum(cluster_sizes):,}
  Mean Cluster Size: {np.mean(cluster_sizes):.1f} cells
  Min/Max Size: {min(cluster_sizes)}/{max(cluster_sizes)} cells
  Total Points: {total_points:,}"""
            
            # 高度和密度统计
            cluster_heights = [cluster['max_height_diff'] for cluster in tower_clusters]
            cluster_densities = [cluster['avg_density'] for cluster in tower_clusters]
            
            stats_text += f"""
            
Quality Metrics:
  Mean Max Height: {np.mean(cluster_heights):.2f} m
  Mean Density: {np.mean(cluster_densities):.2f}
  Height Range: {min(cluster_heights):.1f}-{max(cluster_heights):.1f} m"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_tower_step5_radius_constraint(self, tower_clusters: List[Dict], final_towers: List[Dict],
                                               points: np.ndarray,
                                               title: str = "Tower Step 5: Planar Radius Constraint",
                                               save_name: str = "11_tower_step5_final.png"):
        """可视化塔检测步骤5：平面半径约束和最终结果
        
        Args:
            tower_clusters: 聚类的塔
            final_towers: 最终的塔
            points: 点云数据
            title: 图标题
            save_name: 保存文件名
        """
        print(f"🎨 可视化塔检测步骤5: {len(tower_clusters)} -> {len(final_towers)} 个最终塔")
        
        fig = plt.figure(figsize=(18, 12))
        
        # 背景点云采样
        bg_sample = points[::1000] if len(points) > 10000 else points
        
        # 最终塔的3D视图
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(bg_sample[:, 0], bg_sample[:, 1], bg_sample[:, 2], 
                   c='lightgray', s=1, alpha=0.2)
        
        if final_towers:
            colors = plt.cm.Set1(np.linspace(0, 1, len(final_towers)))
            for i, tower in enumerate(final_towers):
                if 'points' in tower and len(tower['points']) > 0:
                    tower_points = tower['points']
                    ax1.scatter(tower_points[:, 0], tower_points[:, 1], tower_points[:, 2],
                               c=[colors[i]], s=20, label=f'Tower {i}')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Final Towers 3D View ({len(final_towers)} towers)')
        if len(final_towers) <= 5:
            ax1.legend()
        
        # 最终塔的XY平面视图
        ax2 = fig.add_subplot(232)
        ax2.scatter(bg_sample[:, 0], bg_sample[:, 1], c='lightgray', s=1, alpha=0.3)
        
        if final_towers:
            for i, tower in enumerate(final_towers):
                if 'points' in tower and len(tower['points']) > 0:
                    tower_points = tower['points']
                    ax2.scatter(tower_points[:, 0], tower_points[:, 1],
                               c=[colors[i]], s=20, label=f'Tower {i}')
                    
                    # 绘制半径约束圆
                    centroid_2d = tower['centroid'][:2]
                    radius = tower.get('radius', 0)
                    max_radius = tower.get('max_allowed_radius', radius)
                    
                    # 实际半径
                    circle1 = plt.Circle(centroid_2d, radius, fill=False, 
                                       color=colors[i], linestyle='-', alpha=0.8)
                    ax2.add_patch(circle1)
                    
                    # 最大允许半径
                    circle2 = plt.Circle(centroid_2d, max_radius, fill=False, 
                                       color=colors[i], linestyle='--', alpha=0.5)
                    ax2.add_patch(circle2)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Final Towers Top View with Radius Constraints')
        ax2.set_aspect('equal')
        
        # 半径约束分析
        ax3 = fig.add_subplot(233)
        if tower_clusters:
            # 比较通过和未通过半径约束的塔
            failed_towers = [t for t in tower_clusters if t not in final_towers]
            
            if final_towers:
                final_radii = [t.get('radius', 0) for t in final_towers]
                final_max_radii = [t.get('max_allowed_radius', 0) for t in final_towers]
                
                ax3.scatter(final_radii, final_max_radii, c='green', s=50, 
                           alpha=0.7, label=f'Passed ({len(final_towers)})')
            
            # 添加约束线 y=x
            if final_towers:
                max_val = max(max(final_radii), max(final_max_radii))
                ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, 
                        label='Constraint Line (R ≤ r+5)')
            
            ax3.set_xlabel('Actual Radius (m)')
            ax3.set_ylabel('Max Allowed Radius (m)')
            ax3.set_title('Radius Constraint Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 塔的形状规则性分析
        ax4 = fig.add_subplot(234)
        if final_towers:
            aspect_ratios = []
            horizontal_spreads = []
            
            for tower in final_towers:
                if 'points' in tower and len(tower['points']) > 0:
                    tower_points = tower['points']
                    heights = tower_points[:, 2]
                    height_range = heights.max() - heights.min()
                    
                    centroid_2d = tower_points[:, :2].mean(axis=0)
                    horizontal_distances = np.linalg.norm(tower_points[:, :2] - centroid_2d, axis=1)
                    horizontal_spread = np.percentile(horizontal_distances, 95)
                    
                    if horizontal_spread > 0:
                        aspect_ratio = height_range / horizontal_spread
                        aspect_ratios.append(aspect_ratio)
                        horizontal_spreads.append(horizontal_spread)
            
            if aspect_ratios:
                scatter = ax4.scatter(horizontal_spreads, aspect_ratios, 
                                    c=range(len(aspect_ratios)), cmap='plasma', s=60)
                ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, 
                           label='Min Aspect Ratio = 2.0')
                ax4.set_xlabel('Horizontal Spread (m)')
                ax4.set_ylabel('Height/Width Aspect Ratio')
                ax4.set_title('Tower Shape Regularity')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax4, label='Tower Index')
        
        # 塔的高度分布
        ax5 = fig.add_subplot(235)
        if final_towers:
            tower_heights = []
            for tower in final_towers:
                if 'points' in tower and len(tower['points']) > 0:
                    heights = tower['points'][:, 2]
                    tower_heights.extend(heights)
            
            if tower_heights:
                ax5.hist(tower_heights, bins=30, alpha=0.7, color='brown', edgecolor='black')
                ax5.set_xlabel('Height (m)')
                ax5.set_ylabel('Point Count')
                ax5.set_title('Final Tower Height Distribution')
                ax5.grid(True, alpha=0.3)
                
                # 添加统计线
                mean_height = np.mean(tower_heights)
                ax5.axvline(x=mean_height, color='red', linestyle='--', 
                           label=f'Mean: {mean_height:.1f}m')
                ax5.legend()
        
        # 最终统计信息
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        success_rate = len(final_towers) / max(1, len(tower_clusters)) * 100
        
        stats_text = f"""Step 5 Final Results:
        
Tower Clusters: {len(tower_clusters):,}
Final Towers: {len(final_towers):,}
Success Rate: {success_rate:.1f}%"""
        
        if final_towers:
            total_points = sum(len(t.get('points', [])) for t in final_towers)
            tower_radii = [t.get('radius', 0) for t in final_towers]
            
            stats_text += f"""
            
Final Tower Statistics:
  Total Points: {total_points:,}
  Avg Points/Tower: {total_points/len(final_towers):.0f}"""
            
            if tower_radii:
                stats_text += f"""
  Mean Radius: {np.mean(tower_radii):.2f} m
  Radius Range: {min(tower_radii):.1f}-{max(tower_radii):.1f} m"""
            
            # 形状统计
            if 'aspect_ratios' in locals() and aspect_ratios:
                stats_text += f"""
                
Shape Analysis:
  Mean Aspect Ratio: {np.mean(aspect_ratios):.1f}
  Towers with AR>2: {sum(1 for ar in aspect_ratios if ar > 2.0)}"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        if final_towers:
            print(f"  最终提取到 {len(final_towers)} 个塔")
            total_points = sum(len(t.get('points', [])) for t in final_towers)
            print(f"  包含 {total_points:,} 个点 ({total_points/len(points)*100:.2f}%)")

    def visualize_complete_power_grid_system(self, power_lines: List[Dict], towers: List[Dict], 
                                           points: np.ndarray,
                                           title: str = "Complete Power Grid System",
                                           save_name: str = "12_complete_system.png"):
        """可视化完整的电力网格系统（电力线+塔）
        
        Args:
            power_lines: 电力线列表
            towers: 塔列表
            points: 原始点云
            title: 图标题  
            save_name: 保存文件名
        """
        print(f"🎨 可视化完整电力网格系统: {len(power_lines)} 条电力线 + {len(towers)} 个塔")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 背景点云采样
        bg_sample = points[::2000] if len(points) > 20000 else points
        
        # 3D完整系统视图
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(bg_sample[:, 0], bg_sample[:, 1], bg_sample[:, 2], 
                   c='lightgray', s=0.5, alpha=0.1)
        
        # 绘制电力线
        if power_lines:
            line_colors = plt.cm.Blues(np.linspace(0.3, 1, len(power_lines)))
            for i, pl in enumerate(power_lines):
                if 'point_indices' in pl and len(pl['point_indices']) > 0:
                    pl_points = points[pl['point_indices']]
                    ax1.scatter(pl_points[:, 0], pl_points[:, 1], pl_points[:, 2],
                               c=[line_colors[i]], s=10, alpha=0.8, label=f'Line {i}')
        
        # 绘制塔
        if towers:
            tower_colors = plt.cm.Reds(np.linspace(0.3, 1, len(towers)))
            for i, tower in enumerate(towers):
                if 'points' in tower and len(tower['points']) > 0:
                    tower_points = tower['points']
                    ax1.scatter(tower_points[:, 0], tower_points[:, 1], tower_points[:, 2],
                               c=[tower_colors[i]], s=30, alpha=0.9, marker='^', 
                               label=f'Tower {i}')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Complete Power Grid System')
        
        # XY平面视图
        ax2 = fig.add_subplot(232)
        ax2.scatter(bg_sample[:, 0], bg_sample[:, 1], c='lightgray', s=0.5, alpha=0.2)
        
        # 绘制电力线
        if power_lines:
            for i, pl in enumerate(power_lines):
                if 'point_indices' in pl and len(pl['point_indices']) > 0:
                    pl_points = points[pl['point_indices']]
                    ax2.scatter(pl_points[:, 0], pl_points[:, 1],
                               c=[line_colors[i]], s=8, alpha=0.7)
        
        # 绘制塔
        if towers:
            for i, tower in enumerate(towers):
                if 'points' in tower and len(tower['points']) > 0:
                    tower_points = tower['points']
                    ax2.scatter(tower_points[:, 0], tower_points[:, 1],
                               c=[tower_colors[i]], s=40, alpha=0.9, marker='^')
                    
                    # 标记塔的中心
                    centroid_2d = tower['centroid'][:2]
                    ax2.scatter(centroid_2d[0], centroid_2d[1], 
                               c='black', s=100, marker='*', edgecolor='white', linewidth=2)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View - Complete System')
        ax2.set_aspect('equal')
        
        # 高度分析对比
        ax3 = fig.add_subplot(233)
        if power_lines or towers:
            line_heights = []
            tower_heights = []
            
            if power_lines:
                for pl in power_lines:
                    if 'point_indices' in pl and len(pl['point_indices']) > 0:
                        pl_points = points[pl['point_indices']]
                        line_heights.extend(pl_points[:, 2])
            
            if towers:
                for tower in towers:
                    if 'points' in tower and len(tower['points']) > 0:
                        tower_points = tower['points']
                        tower_heights.extend(tower_points[:, 2])
            
            if line_heights:
                ax3.hist(line_heights, bins=30, alpha=0.7, color='blue', 
                        label=f'Power Lines ({len(line_heights)} pts)')
            if tower_heights:
                ax3.hist(tower_heights, bins=30, alpha=0.7, color='red', 
                        label=f'Towers ({len(tower_heights)} pts)')
            
            ax3.set_xlabel('Height (m)')
            ax3.set_ylabel('Point Count')
            ax3.set_title('Height Distribution Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 系统连接分析
        ax4 = fig.add_subplot(234)
        if power_lines and towers:
            # 分析电力线和塔的空间关系
            tower_positions = []
            for tower in towers:
                if 'centroid' in tower:
                    tower_positions.append(tower['centroid'][:2])
            
            if tower_positions:
                tower_positions = np.array(tower_positions)
                
                # 绘制塔的位置
                ax4.scatter(tower_positions[:, 0], tower_positions[:, 1], 
                           c='red', s=100, marker='^', alpha=0.8, label='Towers')
                
                # 绘制电力线的中心线
                for i, pl in enumerate(power_lines):
                    if 'point_indices' in pl and len(pl['point_indices']) > 0:
                        pl_points = points[pl['point_indices']]
                        # 简化为线段的中心轨迹
                        sorted_indices = np.argsort(pl_points[:, 0])  # 按X排序
                        sorted_points = pl_points[sorted_indices]
                        ax4.plot(sorted_points[:, 0], sorted_points[:, 1], 
                                color=line_colors[i], linewidth=2, alpha=0.6)
                
                ax4.set_xlabel('X (m)')
                ax4.set_ylabel('Y (m)')
                ax4.set_title('System Connectivity Analysis')
                ax4.legend()
                ax4.set_aspect('equal')
                ax4.grid(True, alpha=0.3)
        
        # 覆盖范围分析
        ax5 = fig.add_subplot(235)
        if power_lines or towers:
            all_system_points = []
            
            if power_lines:
                for pl in power_lines:
                    if 'point_indices' in pl and len(pl['point_indices']) > 0:
                        all_system_points.extend(points[pl['point_indices']])
            
            if towers:
                for tower in towers:
                    if 'points' in tower and len(tower['points']) > 0:
                        all_system_points.extend(tower['points'])
            
            if all_system_points:
                all_system_points = np.array(all_system_points)
                
                # 计算覆盖范围
                x_range = all_system_points[:, 0].max() - all_system_points[:, 0].min()
                y_range = all_system_points[:, 1].max() - all_system_points[:, 1].min()
                z_range = all_system_points[:, 2].max() - all_system_points[:, 2].min()
                
                coverage_stats = [x_range, y_range, z_range]
                labels = ['X Range (m)', 'Y Range (m)', 'Z Range (m)']
                colors = ['red', 'green', 'blue']
                
                bars = ax5.bar(labels, coverage_stats, color=colors, alpha=0.7)
                ax5.set_ylabel('Range (m)')
                ax5.set_title('System Coverage Analysis')
                ax5.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, val in zip(bars, coverage_stats):
                    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom')
        
        # 系统统计摘要
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        # 计算系统统计
        total_line_points = sum(len(pl.get('point_indices', [])) for pl in power_lines) if power_lines else 0
        total_tower_points = sum(len(t.get('points', [])) for t in towers) if towers else 0
        total_system_points = total_line_points + total_tower_points
        
        total_line_length = sum(pl.get('total_length', 0) for pl in power_lines) if power_lines else 0
        
        stats_text = f"""Complete Power Grid System:
        
POWER LINES:
  Count: {len(power_lines) if power_lines else 0}
  Total Length: {total_line_length:.1f} m
  Points: {total_line_points:,}
  
TOWERS:
  Count: {len(towers) if towers else 0}
  Points: {total_tower_points:,}
  
SYSTEM TOTALS:
  Total Points: {total_system_points:,}
  Coverage: {total_system_points/len(points)*100:.2f}% of input
  
EFFICIENCY:
  Lines/Tower Ratio: {len(power_lines)/max(1, len(towers)):.1f}
  Points/Structure: {total_system_points/max(1, len(power_lines) + len(towers)):.0f}"""
        
        if power_lines and towers:
            # 分析空间分布效率
            line_density = len(power_lines) / max(1, x_range * y_range) * 1000000  # 线/km²
            tower_density = len(towers) / max(1, x_range * y_range) * 1000000     # 塔/km²
            
            stats_text += f"""
            
SPATIAL DENSITY:
  Lines: {line_density:.2f} /km²
  Towers: {tower_density:.2f} /km²
  Coverage Area: {x_range/1000*y_range/1000:.2f} km²"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  完整系统: {len(power_lines)} 条电力线 + {len(towers)} 个塔")
        print(f"  系统点数: {total_system_points:,} ({total_system_points/len(points)*100:.2f}%)")
        if total_line_length > 0:
            print(f"  电力线总长度: {total_line_length:.1f} m")