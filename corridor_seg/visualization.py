"""
Visualization module for corridor segmentation results.
"""

import numpy as np
from typing import Dict, Optional
import logging

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def visualize_results(results: Dict, window_name: str = "Power Transmission Objects") -> None:
    """
    Visualize corridor segmentation results using Open3D.
    
    Args:
        results: Processing results from CorridorSegmenter
        window_name: Window title for visualization
    """
    if not OPEN3D_AVAILABLE:
        logging.warning("Open3D not available. Skipping visualization.")
        return
    
    logger = logging.getLogger(__name__)
    logger.info("Creating 3D visualization")
    
    # Extract point clouds
    original_points = results.get('points_original', np.array([]))
    pl_points = results.get('pl_points', np.array([]))
    tower_points = results.get('tower_points', np.array([]))
    
    if len(original_points) == 0:
        logger.warning("No points to visualize")
        return
    
    # Create visualization geometries
    geometries = []
    
    # 1. Original point cloud (gray, downsampled for performance)
    if len(original_points) > 100000:
        # Downsample for large datasets
        downsample_factor = len(original_points) // 100000
        original_downsampled = original_points[::downsample_factor]
    else:
        original_downsampled = original_points
    
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_downsampled)
    original_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    geometries.append(original_pcd)
    
    # 2. Power line points (red)
    if len(pl_points) > 0:
        pl_pcd = o3d.geometry.PointCloud()
        pl_pcd.points = o3d.utility.Vector3dVector(pl_points)
        pl_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        geometries.append(pl_pcd)
        logger.info(f"Visualizing {len(pl_points)} power line points")
    
    # 3. Tower points (blue)
    if len(tower_points) > 0:
        tower_pcd = o3d.geometry.PointCloud()
        tower_pcd.points = o3d.utility.Vector3dVector(tower_points)
        tower_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        geometries.append(tower_pcd)
        logger.info(f"Visualizing {len(tower_points)} tower points")
    
    # 4. Add bounding boxes for towers
    towers = results.get('towers', [])
    for i, tower in enumerate(towers):
        if 'points' in tower and len(tower['points']) > 0:
            tower_pts = tower['points']
            
            # Create axis-aligned bounding box
            min_bound = tower_pts.min(axis=0)
            max_bound = tower_pts.max(axis=0)
            
            # Create bounding box geometry
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            bbox.color = [0.0, 1.0, 0.0]  # Green
            geometries.append(bbox)
    
    # 5. Add line segments for power lines
    power_lines = results.get('power_lines', [])
    for pl in power_lines:
        endpoints = pl.get('endpoints', [])
        if len(endpoints) == 2:
            # Create line segment
            line_points = np.array(endpoints)
            lines = [[0, 1]]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
            geometries.append(line_set)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    geometries.append(coord_frame)
    
    # Set up visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1200, height=800)
    
    # Add geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set viewing options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    render_option.point_size = 2.0
    render_option.line_width = 3.0
    
    # Set camera to good viewing angle
    view_control = vis.get_view_control()
    if len(original_points) > 0:
        # Center view on point cloud
        centroid = original_points.mean(axis=0)
        bounds = original_points.max(axis=0) - original_points.min(axis=0)
        
        # Set camera position slightly above and offset
        camera_pos = centroid + np.array([bounds[0] * 0.5, -bounds[1] * 0.5, bounds[2] * 0.5])
        view_control.set_lookat(centroid)
        view_control.set_up([0, 0, 1])  # Z-up
    
    logger.info("Visualization window opened. Press 'Q' to close.")
    
    # Add instructions
    print("\n3D Visualization Controls:")
    print("- Mouse: Rotate, zoom, pan")
    print("- 'H': Show help")
    print("- 'Q' or ESC: Close window")
    print("- 'S': Save screenshot")
    print("\nColor Legend:")
    print("- Gray: Original point cloud")
    print("- Red: Power lines")
    print("- Blue: Towers")
    print("- Green: Tower bounding boxes")
    print("- Yellow: Power line connections\n")
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def save_visualization_screenshot(results: Dict, output_path: str, 
                                window_size: tuple = (1200, 800)) -> bool:
    """
    Save a screenshot of the visualization.
    
    Args:
        results: Processing results
        output_path: Output image path
        window_size: Window dimensions (width, height)
        
    Returns:
        success: True if screenshot saved successfully
    """
    if not OPEN3D_AVAILABLE:
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create off-screen visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=window_size[0], height=window_size[1], visible=False)
        
        # Add geometries (simplified version of visualize_results)
        original_points = results.get('points_original', np.array([]))
        pl_points = results.get('pl_points', np.array([]))
        tower_points = results.get('tower_points', np.array([]))
        
        if len(original_points) > 50000:
            downsample_factor = len(original_points) // 50000
            original_downsampled = original_points[::downsample_factor]
        else:
            original_downsampled = original_points
        
        # Original points
        if len(original_downsampled) > 0:
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(original_downsampled)
            original_pcd.paint_uniform_color([0.7, 0.7, 0.7])
            vis.add_geometry(original_pcd)
        
        # Power line points
        if len(pl_points) > 0:
            pl_pcd = o3d.geometry.PointCloud()
            pl_pcd.points = o3d.utility.Vector3dVector(pl_points)
            pl_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            vis.add_geometry(pl_pcd)
        
        # Tower points
        if len(tower_points) > 0:
            tower_pcd = o3d.geometry.PointCloud()
            tower_pcd.points = o3d.utility.Vector3dVector(tower_points)
            tower_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            vis.add_geometry(tower_pcd)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background for screenshots
        render_option.point_size = 3.0
        
        # Set camera view
        view_control = vis.get_view_control()
        if len(original_points) > 0:
            centroid = original_points.mean(axis=0)
            bounds = original_points.max(axis=0) - original_points.min(axis=0)
            
            camera_pos = centroid + np.array([bounds[0] * 0.7, -bounds[1] * 0.7, bounds[2] * 0.5])
            view_control.set_lookat(centroid)
            view_control.set_up([0, 0, 1])
        
        # Render and save
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        
        logger.info(f"Screenshot saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")
        return False


def create_processing_summary_plot(results: Dict, output_path: str) -> bool:
    """
    Create a summary plot of processing statistics.
    
    Args:
        results: Processing results
        output_path: Output image path
        
    Returns:
        success: True if plot saved successfully
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        logging.warning("Matplotlib not available. Skipping summary plot.")
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        stats = results.get('processing_stats', {})
        stage_times = stats.get('stage_times', {})
        point_counts = stats.get('point_counts', {})
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Power Transmission Objects Extraction Summary', fontsize=16)
        
        # 1. Processing time by stage
        if stage_times:
            stages = list(stage_times.keys())
            times = list(stage_times.values())
            
            ax1.bar(stages, times, color='skyblue')
            ax1.set_title('Processing Time by Stage')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Point distribution
        if point_counts:
            labels = list(point_counts.keys())
            counts = list(point_counts.values())
            colors = ['lightgray', 'gray', 'red', 'blue'][:len(labels)]
            
            ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
            ax2.set_title('Point Distribution')
        
        # 3. Detection results
        power_lines = results.get('power_lines', [])
        towers = results.get('towers', [])
        
        categories = ['Power Lines', 'Towers']
        detection_counts = [len(power_lines), len(towers)]
        colors = ['red', 'blue']
        
        bars = ax3.bar(categories, detection_counts, color=colors, alpha=0.7)
        ax3.set_title('Objects Detected')
        ax3.set_ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, detection_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # 4. Processing efficiency
        total_time = stats.get('total_time', 0)
        total_points = point_counts.get('filtered', 0)
        
        if total_time > 0 and total_points > 0:
            points_per_second = total_points / total_time
            ax4.text(0.5, 0.6, f'Processing Speed:\n{points_per_second:.0f} points/sec',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
            ax4.text(0.5, 0.3, f'Total Time: {total_time:.1f}s\nTotal Points: {total_points:,}',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12)
            ax4.set_title('Processing Efficiency')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary plot saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create summary plot: {e}")
        return False