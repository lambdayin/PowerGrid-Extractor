"""
Power line extraction module.
Implements local linear segment detection and global merging with CLF approach.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import logging


class PowerLineExtractor:
    """Power line extraction following Zhang et al. methodology."""
    
    def __init__(self, config):
        """Initialize power line extractor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_local_segments(self, linear_voxels: Dict, voxel_features: Dict, 
                              points: np.ndarray, delta_h_min: float) -> List[Dict]:
        """
        Extract local linear segments from linear voxels.
        
        Args:
            linear_voxels: Voxels identified as having linear structure
            voxel_features: All voxel features
            points: Point cloud
            delta_h_min: Minimum height threshold
            
        Returns:
            segments: List of local PL candidate segments
        """
        self.logger.info("Extracting local power line segments")
        
        segments = []
        segment_id = 0
        
        for voxel_idx, features in linear_voxels.items():
            point_indices = features.get('point_indices', [])
            if len(point_indices) < 3:
                continue
            
            voxel_points = points[point_indices]
            centroid = features['centroid']
            
            # Height filter: only segments above minimum height
            if centroid[2] < delta_h_min:
                continue
            
            # Create segment object
            segment = {
                'segment_id': segment_id,
                'voxel_idx': voxel_idx,
                'point_indices': point_indices,
                'points': voxel_points,
                'centroid': centroid,
                'principal_direction': features['principal_direction'],
                'a1D': features['a1D'],
                'a2D': features['a2D'], 
                'a3D': features['a3D'],
                'eigenvalues': features['eigenvalues'],
                'length': self._estimate_segment_length(voxel_points, features['principal_direction']),
                'azimuth': self._compute_azimuth(features['principal_direction']),
                'height_range': voxel_points[:, 2].max() - voxel_points[:, 2].min()
            }
            
            segments.append(segment)
            segment_id += 1
        
        self.logger.info(f"Extracted {len(segments)} local PL candidate segments")
        return segments
    
    def _estimate_segment_length(self, points: np.ndarray, direction: np.ndarray) -> float:
        """Estimate segment length by projecting onto principal direction."""
        if len(points) < 2:
            return 0.0
        
        # Project points onto principal direction
        centered_points = points - points.mean(axis=0)
        projections = np.dot(centered_points, direction)
        
        return projections.max() - projections.min()
    
    def _compute_azimuth(self, direction: np.ndarray) -> float:
        """Compute azimuth angle in degrees (0-360)."""
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        if azimuth < 0:
            azimuth += 360
        return azimuth
    
    def compute_segment_compatibility(self, seg1: Dict, seg2: Dict) -> Dict:
        """
        Compute compatibility metrics between two segments.
        
        Args:
            seg1, seg2: Segment dictionaries
            
        Returns:
            compatibility: Dictionary with compatibility metrics
        """
        # Distance between centroids
        distance = np.linalg.norm(seg1['centroid'] - seg2['centroid'])
        
        # Height difference
        height_diff = abs(seg1['centroid'][2] - seg2['centroid'][2])
        
        # Direction similarity (collinearity check)
        dir1, dir2 = seg1['principal_direction'], seg2['principal_direction']
        # Handle both parallel and anti-parallel directions
        dot_product = np.abs(np.dot(dir1, dir2))
        angle_diff = np.degrees(np.arccos(np.clip(dot_product, 0, 1)))
        
        # Azimuth difference
        azimuth_diff = abs(seg1['azimuth'] - seg2['azimuth'])
        azimuth_diff = min(azimuth_diff, 360 - azimuth_diff)  # Handle wraparound
        
        return {
            'distance': distance,
            'height_diff': height_diff,
            'angle_diff': angle_diff,
            'azimuth_diff': azimuth_diff,
            'collinear': angle_diff < self.config.collinearity_angle_thr,
            'height_compatible': height_diff < 3.0,  # 3m height tolerance
            'distance_compatible': distance < 50.0   # 50m distance tolerance
        }
    
    def build_segment_graph(self, segments: List[Dict], grid_2d: Dict, 
                            grid_origin: Optional[Tuple[float, float]] = None) -> nx.Graph:
        """
        Build graph connecting compatible segments using 8-neighborhood.
        
        Args:
            segments: Local PL segments
            grid_2d: 2D grid organization
            grid_origin: (min_x, min_y) used by preprocessing grid indexer
            
        Returns:
            graph: NetworkX graph with segment connections
        """
        self.logger.info("Building segment compatibility graph")
        
        # Create segment spatial index
        segment_by_grid = defaultdict(list)
        grid_size = self.config.grid_2d_size
        origin_x = 0.0
        origin_y = 0.0
        if grid_origin is not None:
            try:
                origin_x = float(grid_origin[0])
                origin_y = float(grid_origin[1])
            except Exception:
                pass
        
        for segment in segments:
            centroid = segment['centroid']
            # Map centroid to grid cell
            grid_i = int((centroid[0] - origin_x) // grid_size)
            grid_j = int((centroid[1] - origin_y) // grid_size)
            segment_by_grid[(grid_i, grid_j)].append(segment)
        
        # Build compatibility graph
        graph = nx.Graph()
        
        # Add segments as nodes
        for segment in segments:
            graph.add_node(segment['segment_id'], **segment)
        
        # Check compatibility within neighborhoods
        for (grid_i, grid_j), grid_segments in segment_by_grid.items():
            # Get 8-connected neighbors
            neighbor_cells = self._get_8_neighbors(grid_i, grid_j)
            
            # Check connections within same cell
            for i, seg1 in enumerate(grid_segments):
                for j, seg2 in enumerate(grid_segments[i+1:], i+1):
                    compatibility = self.compute_segment_compatibility(seg1, seg2)
                    if self._is_compatible(compatibility):
                        graph.add_edge(seg1['segment_id'], seg2['segment_id'], **compatibility)
            
            # Check connections to neighboring cells
            for neighbor_cell in neighbor_cells:
                neighbor_segments = segment_by_grid.get(neighbor_cell, [])
                for seg1 in grid_segments:
                    for seg2 in neighbor_segments:
                        compatibility = self.compute_segment_compatibility(seg1, seg2)
                        if self._is_compatible(compatibility):
                            graph.add_edge(seg1['segment_id'], seg2['segment_id'], **compatibility)
        
        self.logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def _get_8_neighbors(self, grid_i: int, grid_j: int) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors of a grid cell."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbors.append((grid_i + di, grid_j + dj))
        return neighbors
    
    def _is_compatible(self, compatibility: Dict) -> bool:
        """Check if two segments are compatible for merging."""
        return (compatibility['collinear'] and 
                compatibility['height_compatible'] and
                compatibility['distance_compatible'])
    
    def merge_segments_global(self, graph: nx.Graph, segments: List[Dict]) -> List[Dict]:
        """
        Perform global segment merging using connected components.
        
        Args:
            graph: Segment compatibility graph
            segments: Local segments
            
        Returns:
            power_lines: Merged power line segments
        """
        self.logger.info("Performing global segment merging")
        
        # Find connected components
        components = list(nx.connected_components(graph))
        power_lines = []
        
        # Create segment lookup
        segment_dict = {seg['segment_id']: seg for seg in segments}
        
        for comp_id, component in enumerate(components):
            if len(component) < 2:  # Skip isolated segments
                continue
            
            # Collect all points from connected segments
            all_points = []
            all_indices = []
            component_segments = []
            
            for seg_id in component:
                segment = segment_dict[seg_id]
                all_points.extend(segment['points'])
                all_indices.extend(segment['point_indices'])
                component_segments.append(segment)
            
            all_points = np.array(all_points)
            
            # Recompute global features for merged line
            merged_features = self._compute_merged_features(all_points, component_segments)
            
            power_line = {
                'powerline_id': comp_id,
                'segment_ids': list(component),
                'point_indices': all_indices,
                'points': all_points,
                'num_segments': len(component),
                'total_length': merged_features['total_length'],
                'centroid': merged_features['centroid'],
                'principal_direction': merged_features['principal_direction'],
                'azimuth': merged_features['azimuth'],
                'height_stats': merged_features['height_stats'],
                'endpoints': merged_features['endpoints'],
                'is_continuous': self._check_continuity(component_segments),
                'component_segments': component_segments
            }
            
            power_lines.append(power_line)
        
        self.logger.info(f"Merged into {len(power_lines)} power line candidates")
        return power_lines
    
    def _compute_merged_features(self, points: np.ndarray, segments: List[Dict]) -> Dict:
        """Compute features for merged power line."""
        # Overall centroid
        centroid = points.mean(axis=0)
        
        # Recompute principal direction using PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(points)
        principal_direction = pca.components_[0]
        
        # Total length estimation
        projections = np.dot(points - centroid, principal_direction)
        total_length = projections.max() - projections.min()
        
        # Height statistics
        heights = points[:, 2]
        height_stats = {
            'mean': heights.mean(),
            'std': heights.std(),
            'min': heights.min(),
            'max': heights.max(),
            'range': heights.max() - heights.min()
        }
        
        # Find endpoints
        min_proj_idx = np.argmin(projections)
        max_proj_idx = np.argmax(projections)
        endpoints = [points[min_proj_idx], points[max_proj_idx]]
        
        return {
            'centroid': centroid,
            'principal_direction': principal_direction,
            'azimuth': self._compute_azimuth(principal_direction),
            'total_length': total_length,
            'height_stats': height_stats,
            'endpoints': endpoints
        }
    
    def _check_continuity(self, segments: List[Dict]) -> bool:
        """Check if segments form a continuous line."""
        if len(segments) < 2:
            return True
        
        # Simple continuity check based on spatial gaps
        centroids = np.array([seg['centroid'] for seg in segments])
        distances = cdist(centroids, centroids)
        
        # Check if all segments are within reasonable distance of at least one other
        max_gap = self.config.grid_2d_size * 3  # 3 grid cells
        for i in range(len(segments)):
            min_dist = np.min(distances[i][distances[i] > 0])  # Exclude self-distance
            if min_dist > max_gap:
                return False
        
        return True
    
    def filter_power_lines(self, power_lines: List[Dict], min_length: float = 10.0,
                          min_segments: int = 2) -> List[Dict]:
        """
        Filter power lines based on geometric criteria.
        
        Args:
            power_lines: Candidate power lines
            min_length: Minimum line length
            min_segments: Minimum number of segments
            
        Returns:
            filtered_lines: Filtered power lines
        """
        self.logger.info(f"Filtering power lines (min_length={min_length}m, min_segments={min_segments})")
        
        filtered_lines = []
        
        for pl in power_lines:
            # Length filter
            if pl['total_length'] < min_length:
                continue
            
            # Segment count filter  
            if pl['num_segments'] < min_segments:
                continue
            
            # Continuity filter
            if not pl['is_continuous']:
                continue
            
            # Height consistency filter
            height_std = pl['height_stats']['std']
            if height_std > 10.0:  # Too much height variation
                continue
            
            filtered_lines.append(pl)
        
        self.logger.info(f"Filtered to {len(filtered_lines)} valid power lines")
        return filtered_lines
    
    def extract_power_lines(self, linear_voxels: Dict, voxel_features: Dict,
                           points: np.ndarray, grid_2d: Dict, delta_h_min: float,
                           grid_origin: Optional[Tuple[float, float]] = None) -> Tuple[List[Dict], np.ndarray]:
        """
        Complete power line extraction pipeline.
        
        Args:
            linear_voxels: Linear structure voxels
            voxel_features: All voxel features  
            points: Point cloud
            grid_2d: 2D grid organization
            delta_h_min: Minimum height threshold
            grid_origin: (min_x, min_y) used by preprocessing grid indexer
            
        Returns:
            power_lines: Extracted power lines
            pl_mask: Boolean mask indicating power line points
        """
        self.logger.info("Starting complete power line extraction")
        
        # 1. Extract local segments
        segments = self.extract_local_segments(linear_voxels, voxel_features, points, delta_h_min)
        
        if not segments:
            self.logger.warning("No local segments found")
            return [], np.zeros(len(points), dtype=bool)
        
        # 2. Build compatibility graph
        graph = self.build_segment_graph(segments, grid_2d, grid_origin)
        
        # 3. Global merging
        power_lines = self.merge_segments_global(graph, segments)
        
        # 4. Filter results
        power_lines = self.filter_power_lines(power_lines)
        
        # 5. Create point mask
        pl_mask = np.zeros(len(points), dtype=bool)
        for pl in power_lines:
            pl_mask[pl['point_indices']] = True
        
        self.logger.info(f"Power line extraction complete: {len(power_lines)} lines, "
                        f"{pl_mask.sum()} points ({pl_mask.sum()/len(points)*100:.1f}%)")
        
        return power_lines, pl_mask