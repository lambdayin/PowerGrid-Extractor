"""
Tower candidate region segmentation module.
Implements the 5-step tower detection process from Zhang et al.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.ndimage import label, binary_opening
from sklearn.cluster import DBSCAN
import cv2
import logging


class TowerExtractor:
    """Tower extraction following Zhang et al. 5-step methodology."""
    
    def __init__(self, config):
        """Initialize tower extractor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def step1_height_diff_initial_screening(self, grid_features: Dict, 
                                          delta_h_min: float, tower_head_height: float) -> Set[Tuple[int, int]]:
        """
        Step 1: Initial screening based on large height difference grids.
        
        Args:
            grid_features: 2D grid features with HeightDiff
            delta_h_min: Minimum height gap
            tower_head_height: Tower head height estimate
            
        Returns:
            candidate_grids: Set of grid indices with large height differences
        """
        self.logger.info("Step 1: Initial height difference screening")
        
        candidate_grids = set()
        threshold = max(delta_h_min, tower_head_height * 0.5)
        
        for grid_idx, features in grid_features.items():
            height_diff = features.get('HeightDiff', 0)
            
            # Large height difference criteria
            if height_diff > threshold:
                candidate_grids.add(grid_idx)
        
        self.logger.info(f"Step 1: Found {len(candidate_grids)} candidate grids with HeightDiff > {threshold:.1f}m")
        return candidate_grids
    
    def step2_moving_window_refinement(self, grid_features: Dict, candidate_grids: Set,
                                     tower_head_height: float) -> Set[Tuple[int, int]]:
        """
        Step 2: 2×2 moving window refinement.
        
        Args:
            grid_features: 2D grid features
            candidate_grids: Initial candidate grids from step 1
            tower_head_height: Tower head height estimate
            
        Returns:
            refined_candidates: Refined candidate grids
        """
        self.logger.info("Step 2: 2×2 moving window refinement")
        
        if not candidate_grids:
            return set()
        
        # Create grid bounds
        all_grid_indices = list(grid_features.keys())
        if not all_grid_indices:
            return set()
        
        min_i = min(idx[0] for idx in all_grid_indices)
        max_i = max(idx[0] for idx in all_grid_indices)
        min_j = min(idx[1] for idx in all_grid_indices)
        max_j = max(idx[1] for idx in all_grid_indices)
        
        refined_candidates = set()
        window_size = self.config.moving_window_size
        threshold = tower_head_height + self.config.tower_grid_cluster_offset
        
        # Apply 2×2 moving window
        for i in range(min_i, max_i - window_size + 2):
            for j in range(min_j, max_j - window_size + 2):
                
                # Define 2×2 window
                window_cells = []
                window_heights = []
                
                for di in range(window_size):
                    for dj in range(window_size):
                        cell_idx = (i + di, j + dj)
                        if cell_idx in grid_features:
                            window_cells.append(cell_idx)
                            window_heights.append(grid_features[cell_idx].get('HeightDiff', 0))
                
                if not window_heights:
                    continue
                
                # Check if window has significant height variation
                max_height = max(window_heights)
                height_range = max(window_heights) - min(window_heights)
                
                if max_height > threshold or height_range > tower_head_height:
                    # Add cells that contribute to height variation
                    for cell_idx in window_cells:
                        if grid_features[cell_idx].get('HeightDiff', 0) > tower_head_height:
                            refined_candidates.add(cell_idx)
        
        # Keep intersection with initial candidates
        refined_candidates = refined_candidates.intersection(candidate_grids)
        
        self.logger.info(f"Step 2: Refined to {len(refined_candidates)} candidate grids")
        return refined_candidates
    
    def step3_vertical_continuity_check(self, refined_candidates: Set, grid_features: Dict,
                                      points: np.ndarray, tower_head_height: float,
                                      grid_2d: Optional[Dict] = None,
                                      grid_origin: Optional[Tuple[float, float]] = None) -> Set[Tuple[int, int]]:
        """
        Step 3: Vertical continuity and height threshold check.
        
        Args:
            refined_candidates: Candidate grids from step 2
            grid_features: Grid features
            points: Point cloud
            tower_head_height: Tower head height estimate
            
        Returns:
            valid_candidates: Candidates passing vertical continuity check
        """
        self.logger.info("Step 3: Vertical continuity check")
        
        valid_candidates = set()
        max_height_gap = tower_head_height  # Maximum allowed vertical gap
        
        for grid_idx in refined_candidates:
            if grid_idx not in grid_features:
                continue
            
            features = grid_features[grid_idx]
            
            # Get points in this grid cell using preprocessing grid if available
            if grid_2d is not None and grid_idx in grid_2d:
                point_indices = grid_2d[grid_idx]
                if not point_indices:
                    continue
                grid_points = points[point_indices]
            else:
                # Fallback: spatial bounds using provided origin (less reliable)
                grid_size = self.config.grid_2d_size
                origin_x, origin_y = 0.0, 0.0
                if grid_origin is not None:
                    try:
                        origin_x = float(grid_origin[0])
                        origin_y = float(grid_origin[1])
                    except Exception:
                        pass
                grid_i, grid_j = grid_idx
                x_min = origin_x + grid_i * grid_size
                x_max = origin_x + (grid_i + 1) * grid_size
                y_min = origin_y + grid_j * grid_size
                y_max = origin_y + (grid_j + 1) * grid_size
                mask = ((points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                       (points[:, 1] >= y_min) & (points[:, 1] < y_max))
                if not np.any(mask):
                    continue
                grid_points = points[mask]
            heights = grid_points[:, 2]
            
            # Check vertical continuity
            if self._check_vertical_continuity(heights, max_height_gap):
                # Check if maximum height exceeds threshold
                if heights.max() > max_height_gap:
                    valid_candidates.add(grid_idx)
        
        self.logger.info(f"Step 3: {len(valid_candidates)} candidates pass vertical continuity check")
        return valid_candidates
    
    def _check_vertical_continuity(self, heights: np.ndarray, max_gap: float) -> bool:
        """Check if heights form a vertically continuous distribution."""
        if len(heights) < 3:
            return False
        
        # Sort heights
        sorted_heights = np.sort(heights)
        
        # Check for large vertical gaps
        height_diffs = np.diff(sorted_heights)
        max_diff = np.max(height_diffs)
        
        return max_diff < max_gap
    
    def step4_clustering_to_towers(self, valid_candidates: Set, grid_features: Dict) -> List[Dict]:
        """
        Step 4: Cluster adjacent candidate grids into single towers.
        
        Args:
            valid_candidates: Valid candidate grids from step 3
            grid_features: Grid features
            
        Returns:
            tower_clusters: List of tower cluster dictionaries
        """
        self.logger.info("Step 4: Clustering candidate grids into towers")
        
        if not valid_candidates:
            return []
        
        # Build adjacency graph for candidate grids
        candidate_list = list(valid_candidates)
        adjacency = defaultdict(set)
        
        for i, grid1 in enumerate(candidate_list):
            for j, grid2 in enumerate(candidate_list[i+1:], i+1):
                # Check if grids are adjacent (8-connected)
                if self._are_adjacent_grids(grid1, grid2):
                    adjacency[grid1].add(grid2)
                    adjacency[grid2].add(grid1)
        
        # Find connected components using DFS
        visited = set()
        tower_clusters = []
        cluster_id = 0
        
        for grid_idx in candidate_list:
            if grid_idx in visited:
                continue
            
            # DFS to find connected component
            cluster_grids = set()
            stack = [grid_idx]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster_grids.add(current)
                
                # Add unvisited neighbors
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            # Create tower cluster
            if len(cluster_grids) > 0:
                tower_cluster = self._create_tower_cluster(cluster_id, cluster_grids, grid_features)
                tower_clusters.append(tower_cluster)
                cluster_id += 1
        
        self.logger.info(f"Step 4: Formed {len(tower_clusters)} tower clusters")
        return tower_clusters
    
    def _are_adjacent_grids(self, grid1: Tuple[int, int], grid2: Tuple[int, int]) -> bool:
        """Check if two grid cells are 8-connected neighbors."""
        di = abs(grid1[0] - grid2[0])
        dj = abs(grid1[1] - grid2[1])
        return di <= 1 and dj <= 1 and (di + dj) > 0
    
    def _create_tower_cluster(self, cluster_id: int, grid_cells: Set, grid_features: Dict) -> Dict:
        """Create tower cluster dictionary from grid cells."""
        # Aggregate features from constituent grid cells
        total_points = 0
        height_diffs = []
        centroids = []
        densities = []
        
        for grid_idx in grid_cells:
            if grid_idx in grid_features:
                features = grid_features[grid_idx]
                total_points += features.get('point_count', 0)
                height_diffs.append(features.get('HeightDiff', 0))
                centroids.append(features.get('centroid', np.zeros(3)))
                densities.append(features.get('density', 0))
        
        # Compute cluster statistics
        avg_centroid = np.mean(centroids, axis=0) if centroids else np.zeros(3)
        max_height_diff = max(height_diffs) if height_diffs else 0
        total_density = sum(densities)
        
        return {
            'cluster_id': cluster_id,
            'grid_cells': grid_cells,
            'num_cells': len(grid_cells),
            'total_points': total_points,
            'centroid': avg_centroid,
            'max_height_diff': max_height_diff,
            'total_density': total_density,
            'avg_density': total_density / len(grid_cells) if grid_cells else 0
        }
    
    def step5_planar_radius_constraint(self, tower_clusters: List[Dict], 
                                     points: np.ndarray,
                                     grid_2d: Optional[Dict] = None,
                                     grid_origin: Optional[Tuple[float, float]] = None) -> List[Dict]:
        """
        Step 5: Apply planar projection radius constraint R(m,n) ≤ r+5.
        
        Args:
            tower_clusters: Tower clusters from step 4
            points: Point cloud
            
        Returns:
            filtered_towers: Towers passing radius constraint
        """
        self.logger.info("Step 5: Applying planar radius constraint")
        
        filtered_towers = []
        grid_size = self.config.grid_2d_size
        radius_offset = self.config.planar_radius_offset
        
        for cluster in tower_clusters:
            # Estimate tower footprint from grid cells
            grid_cells = cluster['grid_cells']
            
            # Get all points belonging to this tower cluster
            cluster_points = self._get_cluster_points(cluster, points, grid_size, grid_2d, grid_origin)
            
            if len(cluster_points) < 10:  # Minimum points for valid tower
                continue
            
            # Compute planar projection radius
            centroid_2d = cluster['centroid'][:2]  # X, Y only
            cluster_points_2d = cluster_points[:, :2]
            
            distances = np.linalg.norm(cluster_points_2d - centroid_2d, axis=1)
            radius = np.percentile(distances, 95)  # Use 95th percentile to handle outliers
            
            # Estimate tower wingspan (r)
            # Use heuristic based on height and cell count
            estimated_wingspan = max(5.0, min(15.0, len(grid_cells) * 2))  # 5-15m range
            
            # Apply constraint R(m,n) ≤ r + 5
            max_allowed_radius = estimated_wingspan + radius_offset
            
            if radius <= max_allowed_radius:
                # Add additional tower properties
                cluster['radius'] = radius
                cluster['estimated_wingspan'] = estimated_wingspan
                cluster['max_allowed_radius'] = max_allowed_radius
                cluster['points'] = cluster_points
                cluster['passes_radius_constraint'] = True
                
                # Additional filtering based on shape regularity
                if self._check_shape_regularity(cluster_points):
                    filtered_towers.append(cluster)
            
        self.logger.info(f"Step 5: {len(filtered_towers)} towers pass radius constraint")
        return filtered_towers
    
    def _get_cluster_points(self, cluster: Dict, points: np.ndarray, grid_size: float,
                            grid_2d: Optional[Dict] = None,
                            grid_origin: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Get all points belonging to a tower cluster."""
        cluster_points = []
        
        if grid_2d is not None:
            for grid_idx in cluster['grid_cells']:
                if grid_idx in grid_2d:
                    indices = grid_2d[grid_idx]
                    if indices:
                        cluster_points.extend(points[indices])
            return np.array(cluster_points) if cluster_points else np.empty((0, 3))
        
        # Fallback to spatial bounds using origin
        origin_x, origin_y = 0.0, 0.0
        if grid_origin is not None:
            try:
                origin_x = float(grid_origin[0])
                origin_y = float(grid_origin[1])
            except Exception:
                pass
        
        for grid_idx in cluster['grid_cells']:
            grid_i, grid_j = grid_idx
            x_min = origin_x + grid_i * grid_size
            x_max = origin_x + (grid_i + 1) * grid_size
            y_min = origin_y + grid_j * grid_size
            y_max = origin_y + (grid_j + 1) * grid_size
            mask = ((points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                   (points[:, 1] >= y_min) & (points[:, 1] < y_max))
            if np.any(mask):
                cluster_points.extend(points[mask])
        
        return np.array(cluster_points) if cluster_points else np.empty((0, 3))
    
    def _check_shape_regularity(self, points: np.ndarray) -> bool:
        """Check if tower points form a regular vertical structure."""
        if len(points) < 10:
            return False
        
        # Check vertical extent vs horizontal spread
        heights = points[:, 2]
        height_range = heights.max() - heights.min()
        
        # Check horizontal spread
        centroid_2d = points[:, :2].mean(axis=0)
        horizontal_distances = np.linalg.norm(points[:, :2] - centroid_2d, axis=1)
        horizontal_spread = np.percentile(horizontal_distances, 95)
        
        # Tower should be tall and narrow
        aspect_ratio = height_range / max(horizontal_spread, 1.0)
        
        return aspect_ratio > 2.0  # At least 2:1 height to width ratio
    
    def extract_tower_candidates(self, grid_features: Dict, points: np.ndarray,
                               delta_h_min: float, tower_head_height: float,
                               grid_2d: Optional[Dict] = None,
                               grid_origin: Optional[Tuple[float, float]] = None) -> Tuple[List[Dict], np.ndarray]:
        """
        Complete 5-step tower candidate extraction pipeline.
        
        Args:
            grid_features: 2D grid features
            points: Point cloud
            delta_h_min: Minimum height gap
            tower_head_height: Tower head height estimate
            
        Returns:
            towers: Extracted tower candidates
            tower_mask: Boolean mask indicating tower points
        """
        self.logger.info("Starting complete 5-step tower extraction")
        
        # Step 1: Initial screening
        candidates_step1 = self.step1_height_diff_initial_screening(
            grid_features, delta_h_min, tower_head_height)
        
        if not candidates_step1:
            self.logger.warning("No candidates after step 1")
            return [], np.zeros(len(points), dtype=bool)
        
        # Step 2: Moving window refinement
        candidates_step2 = self.step2_moving_window_refinement(
            grid_features, candidates_step1, tower_head_height)
        
        if not candidates_step2:
            self.logger.warning("No candidates after step 2")
            return [], np.zeros(len(points), dtype=bool)
        
        # Step 3: Vertical continuity check
        candidates_step3 = self.step3_vertical_continuity_check(
            candidates_step2, grid_features, points, tower_head_height,
            grid_2d=grid_2d, grid_origin=grid_origin)
        
        if not candidates_step3:
            self.logger.warning("No candidates after step 3")
            return [], np.zeros(len(points), dtype=bool)
        
        # Step 4: Clustering to towers
        tower_clusters = self.step4_clustering_to_towers(candidates_step3, grid_features)
        
        if not tower_clusters:
            self.logger.warning("No tower clusters after step 4")
            return [], np.zeros(len(points), dtype=bool)
        
        # Step 5: Planar radius constraint
        towers = self.step5_planar_radius_constraint(tower_clusters, points,
                                                    grid_2d=grid_2d,
                                                    grid_origin=grid_origin)
        
        # Create point mask
        tower_mask = np.zeros(len(points), dtype=bool)
        for tower in towers:
            if 'points' in tower and len(tower['points']) > 0:
                # Find indices of tower points in original cloud
                tower_points = tower['points']
                for tp in tower_points:
                    # Find closest point in original cloud (not ideal but functional)
                    distances = np.sum((points - tp)**2, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 0.1:  # 10cm tolerance
                        tower_mask[closest_idx] = True
        
        self.logger.info(f"Tower extraction complete: {len(towers)} towers, "
                        f"{tower_mask.sum()} points ({tower_mask.sum()/len(points)*100:.1f}%)")
        
        return towers, tower_mask