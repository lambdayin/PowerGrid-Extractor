"""
Preprocessing module for point cloud data.
Implements statistical outlier removal, 2D grid organization, and 3D voxel hashing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import open3d as o3d
from scipy.spatial import KDTree
import logging


class PointCloudPreprocessor:
    """Point cloud preprocessing following Zhang et al. methodology."""
    
    def __init__(self, config):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def statistical_outlier_removal(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Statistical Outlier Removal (SOR) filtering.
        
        Args:
            points: Input point cloud (N, 3)
            
        Returns:
            filtered_points: Cleaned point cloud
            inlier_mask: Boolean mask of inlier points
        """
        self.logger.info(f"Applying SOR with k={self.config.sor_k}, sigma={self.config.sor_sigma}")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Apply statistical outlier removal
        pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=self.config.sor_k,
            std_ratio=self.config.sor_sigma
        )
        
        # Convert back to numpy
        filtered_points = np.asarray(pcd_filtered.points)
        inlier_mask = np.zeros(len(points), dtype=bool)
        inlier_mask[inlier_indices] = True
        
        self.logger.info(f"SOR removed {len(points) - len(filtered_points)} outliers "
                        f"({(1 - len(filtered_points)/len(points))*100:.1f}%)")
        
        return filtered_points, inlier_mask
    
    def create_2d_grid(self, points: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
        """
        Organize points into 2D grid cells (5m x 5m as per paper).
        
        Args:
            points: Point cloud (N, 3)
            
        Returns:
            grid: Dictionary mapping (i, j) grid indices to point indices
        """
        grid = defaultdict(list)
        grid_size = self.config.grid_2d_size
        
        # Calculate grid bounds
        min_x, min_y = points[:, 0].min(), points[:, 1].min()
        
        # Assign points to grid cells
        for idx, point in enumerate(points):
            x, y = point[0], point[1]
            grid_i = int((x - min_x) // grid_size)
            grid_j = int((y - min_y) // grid_size)
            grid[(grid_i, grid_j)].append(idx)
        
        self.logger.info(f"Created 2D grid with {len(grid)} occupied cells, "
                        f"grid size: {grid_size}m x {grid_size}m")
        
        return dict(grid)
    
    def create_3d_voxel_hash(self, points: np.ndarray) -> Dict[Tuple[int, int, int], List[int]]:
        """
        Create 3D voxel hash table for spatial organization (0.5m³ as per paper).
        
        Args:
            points: Point cloud (N, 3)
            
        Returns:
            voxel_hash: Dictionary mapping (i, j, k) voxel indices to point indices
        """
        voxel_hash = defaultdict(list)
        voxel_size = self.config.voxel_size
        
        # Calculate voxel bounds
        min_coords = points.min(axis=0)
        
        # Assign points to voxels
        for idx, point in enumerate(points):
            voxel_indices = tuple(((point - min_coords) // voxel_size).astype(int))
            voxel_hash[voxel_indices].append(idx)
        
        self.logger.info(f"Created 3D voxel hash with {len(voxel_hash)} occupied voxels, "
                        f"voxel size: {voxel_size}m³")
        
        return dict(voxel_hash)
    
    def get_8_neighbors(self, grid_i: int, grid_j: int) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors of a 2D grid cell."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbors.append((grid_i + di, grid_j + dj))
        return neighbors
    
    def get_26_neighbors(self, voxel_i: int, voxel_j: int, voxel_k: int) -> List[Tuple[int, int, int]]:
        """Get 26-connected neighbors of a 3D voxel."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    neighbors.append((voxel_i + di, voxel_j + dj, voxel_k + dk))
        return neighbors
    
    def compute_height_histogram(self, points: np.ndarray, num_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute height histogram for automatic parameter estimation.
        
        Args:
            points: Point cloud (N, 3)
            num_bins: Number of histogram bins
            
        Returns:
            hist: Histogram counts
            bin_edges: Histogram bin edges
        """
        heights = points[:, 2]  # Z coordinates
        hist, bin_edges = np.histogram(heights, bins=num_bins)
        
        self.logger.info(f"Height range: {heights.min():.2f}m to {heights.max():.2f}m")
        
        return hist, bin_edges
    
    def estimate_height_parameters(self, points: np.ndarray) -> Tuple[float, float]:
        """
        Estimate Δh_min and H_h from height histogram.
        
        Args:
            points: Point cloud (N, 3)
            
        Returns:
            delta_h_min: Minimum height gap
            tower_head_height: Tower head height estimate
        """
        hist, bin_edges = self.compute_height_histogram(points)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks in histogram
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=len(points) * 0.001)  # At least 0.1% of points
        
        if len(peaks) >= 2:
            # First and last significant peaks
            first_peak_height = bin_centers[peaks[0]]
            last_peak_height = bin_centers[peaks[-1]]
            tower_head_height = last_peak_height - first_peak_height
        else:
            # Fallback to range-based estimation
            height_range = points[:, 2].max() - points[:, 2].min()
            tower_head_height = height_range * 0.8
        
        # Use default or estimated delta_h_min
        delta_h_min = self.config.min_height_gap
        
        self.logger.info(f"Estimated parameters - Δh_min: {delta_h_min:.2f}m, "
                        f"H_h: {tower_head_height:.2f}m")
        
        return delta_h_min, tower_head_height
    
    def preprocess(self, points: np.ndarray) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            points: Raw point cloud (N, 3)
            
        Returns:
            preprocessed_data: Dictionary containing all preprocessed data
        """
        self.logger.info("Starting point cloud preprocessing")
        
        # 1. Statistical Outlier Removal
        filtered_points, inlier_mask = self.statistical_outlier_removal(points)
        
        # 2. Create 2D grid organization
        grid_2d = self.create_2d_grid(filtered_points)
        
        # 3. Create 3D voxel hash
        voxel_hash_3d = self.create_3d_voxel_hash(filtered_points)
        
        # 4. Estimate height parameters
        delta_h_min, tower_head_height = self.estimate_height_parameters(filtered_points)
        
        # Package results
        preprocessed_data = {
            'points': filtered_points,
            'original_points': points,
            'inlier_mask': inlier_mask,
            'grid_2d': grid_2d,
            'voxel_hash_3d': voxel_hash_3d,
            'delta_h_min': delta_h_min,
            'tower_head_height': tower_head_height,
            'bounds': {
                'min_coords': filtered_points.min(axis=0),
                'max_coords': filtered_points.max(axis=0),
                'grid_size': self.config.grid_2d_size,
                'voxel_size': self.config.voxel_size
            }
        }
        
        self.logger.info("Preprocessing completed successfully")
        return preprocessed_data