"""
Feature calculation module.
Implements 3D dimensional features and 2D distribution features.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import logging


class FeatureCalculator:
    """Feature calculation following Zhang et al. methodology."""
    
    def __init__(self, config):
        """Initialize feature calculator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_covariance_features(self, points: np.ndarray) -> Dict[str, float]:
        """
        Compute 3D dimensional features based on covariance eigenvalues.
        
        Following paper formulation:
        a1D = (√λ1 - √λ2) / √λ1  (linearity)
        a2D = (√λ2 - √λ3) / √λ1  (planarity) 
        a3D = √λ3 / √λ1           (sphericity)
        
        where λ1 ≥ λ2 ≥ λ3 are eigenvalues of covariance matrix
        
        Args:
            points: Local point cluster (N, 3)
            
        Returns:
            features: Dictionary with dimensional features
        """
        if len(points) < 3:
            return {'a1D': 0.0, 'a2D': 0.0, 'a3D': 0.0, 'eigenvalues': [0, 0, 0]}
        
        # Center the points
        centered_points = points - points.mean(axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Ensure non-negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Calculate dimensional features
        sqrt_lambdas = np.sqrt(eigenvalues)
        lambda1, lambda2, lambda3 = sqrt_lambdas
        
        a1D = (lambda1 - lambda2) / lambda1 if lambda1 > 0 else 0.0  # Linearity
        a2D = (lambda2 - lambda3) / lambda1 if lambda1 > 0 else 0.0  # Planarity
        a3D = lambda3 / lambda1 if lambda1 > 0 else 0.0              # Sphericity
        
        return {
            'a1D': a1D,
            'a2D': a2D, 
            'a3D': a3D,
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors,
            'principal_direction': eigenvectors[:, 0]  # Direction of largest eigenvalue
        }
    
    def compute_voxel_features(self, voxel_hash: Dict, points: np.ndarray) -> Dict:
        """
        Compute 3D dimensional features for each voxel.
        
        Args:
            voxel_hash: 3D voxel hash table
            points: Point cloud
            
        Returns:
            voxel_features: Dictionary mapping voxel indices to features
        """
        self.logger.info("Computing 3D dimensional features for voxels")
        
        voxel_features = {}
        
        for voxel_idx, point_indices in voxel_hash.items():
            if len(point_indices) < 3:
                continue
                
            voxel_points = points[point_indices]
            features = self.compute_covariance_features(voxel_points)
            
            # Add voxel-specific information
            features.update({
                'point_count': len(point_indices),
                'point_indices': point_indices,
                'centroid': voxel_points.mean(axis=0),
                'height_range': voxel_points[:, 2].max() - voxel_points[:, 2].min()
            })
            
            voxel_features[voxel_idx] = features
        
        self.logger.info(f"Computed features for {len(voxel_features)} voxels")
        return voxel_features
    
    def compute_2d_grid_features(self, grid_2d: Dict, points: np.ndarray, 
                                pl_candidate_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Compute 2D distribution features for grid cells.
        
        IMPORTANT: When pl_candidate_mask is provided, DSM/nDSM are computed 
        AFTER removing PL candidate points (as specified in paper Section 3.2).
        
        Args:
            grid_2d: 2D grid hash table
            points: Point cloud
            pl_candidate_mask: Boolean mask indicating PL candidate points to exclude
            
        Returns:
            grid_features: Dictionary mapping grid indices to 2D features
        """
        self.logger.info("Computing 2D distribution features for grid cells")
        
        if pl_candidate_mask is not None:
            self.logger.info("Computing DSM features AFTER removing PL candidate points")
            # Use only non-PL points for DSM calculation
            dsm_points = points[~pl_candidate_mask]
        else:
            dsm_points = points
            
        grid_features = {}
        
        for grid_idx, point_indices in grid_2d.items():
            if len(point_indices) == 0:
                continue
            
            grid_points = points[point_indices]
            heights = grid_points[:, 2]
            
            # Basic statistics
            features = {
                'point_count': len(point_indices),
                'density': len(point_indices),  # Points per cell
                'height_mean': heights.mean(),
                'height_std': heights.std() if len(heights) > 1 else 0.0,
                'height_range': heights.max() - heights.min()
            }
            
            # DEM (Digital Elevation Model) - ground points minimum
            features['DEM'] = heights.min()
            
            # DSM (Digital Surface Model) - using filtered points if PL mask provided
            if pl_candidate_mask is not None:
                # Find DSM points in this grid cell (excluding PL candidates)
                non_pl_indices = [idx for idx in point_indices if not pl_candidate_mask[idx]]
                if non_pl_indices:
                    dsm_heights = points[non_pl_indices, 2]
                    features['DSM'] = dsm_heights.max()
                else:
                    features['DSM'] = heights.max()  # Fallback to all points
            else:
                features['DSM'] = heights.max()
            
            # nDSM (normalized DSM)
            features['nDSM'] = features['DSM'] - features['DEM']
            
            # HeightDiff = DSM - DEM
            features['HeightDiff'] = features['nDSM']
            
            # Height percentiles
            features['height_p10'] = np.percentile(heights, 10)
            features['height_p50'] = np.percentile(heights, 50)
            features['height_p90'] = np.percentile(heights, 90)
            
            # Grid centroid
            features['centroid'] = grid_points.mean(axis=0)
            
            grid_features[grid_idx] = features
        
        self.logger.info(f"Computed 2D features for {len(grid_features)} grid cells")
        return grid_features
    
    def identify_linear_structures(self, voxel_features: Dict) -> Dict[Tuple, Dict]:
        """
        Identify linear structures based on dimensional features.
        
        Args:
            voxel_features: Voxel features dictionary
            
        Returns:
            linear_voxels: Dictionary of voxels with high linearity
        """
        self.logger.info("Identifying linear structures from dimensional features")
        
        linear_voxels = {}
        linear_threshold = self.config.a1d_linear_thr
        
        for voxel_idx, features in voxel_features.items():
            a1D = features.get('a1D', 0)
            a2D = features.get('a2D', 0)
            a3D = features.get('a3D', 0)
            
            # Linear structure criteria: high a1D, low a2D and a3D
            if (a1D > linear_threshold and 
                a2D < self.config.a2d_planar_thr and 
                a3D < self.config.a3d_spherical_thr):
                
                linear_voxels[voxel_idx] = features
        
        self.logger.info(f"Identified {len(linear_voxels)} linear structure voxels")
        return linear_voxels
    
    def compute_height_based_features(self, grid_features: Dict, 
                                    delta_h_min: float, tower_head_height: float) -> Dict:
        """
        Compute height-based features for tower candidate identification.
        
        Args:
            grid_features: 2D grid features
            delta_h_min: Minimum height gap
            tower_head_height: Tower head height estimate
            
        Returns:
            height_features: Enhanced grid features with height analysis
        """
        enhanced_features = {}
        
        for grid_idx, features in grid_features.items():
            enhanced = features.copy()
            
            # Height gap analysis
            enhanced['above_min_height'] = features['HeightDiff'] > delta_h_min
            enhanced['tower_height_candidate'] = features['HeightDiff'] > tower_head_height
            enhanced['high_height_diff'] = features['HeightDiff'] > (tower_head_height + 5.0)
            
            # Relative height features
            enhanced['height_ratio'] = features['nDSM'] / max(tower_head_height, 1.0)
            enhanced['density_height_product'] = features['density'] * features['nDSM']
            
            enhanced_features[grid_idx] = enhanced
        
        return enhanced_features
    
    def compute_compass_line_features(self, linear_voxels: Dict, points: np.ndarray) -> Dict:
        """
        Compute Compass Line Filter (CLF) inspired direction features.
        
        Args:
            linear_voxels: Linear structure voxels
            points: Point cloud
            
        Returns:
            direction_features: Direction analysis for each linear structure
        """
        direction_features = {}
        
        for voxel_idx, features in linear_voxels.items():
            principal_dir = features.get('principal_direction', np.array([1, 0, 0]))
            
            # Compute azimuth angle (0-360 degrees)
            azimuth = np.degrees(np.arctan2(principal_dir[1], principal_dir[0]))
            if azimuth < 0:
                azimuth += 360
            
            # Compute inclination angle  
            inclination = np.degrees(np.arcsin(np.abs(principal_dir[2])))
            
            direction_features[voxel_idx] = {
                'azimuth': azimuth,
                'inclination': inclination,
                'principal_direction': principal_dir,
                'is_horizontal': inclination < 15,  # Nearly horizontal structures
                'direction_strength': features.get('a1D', 0)
            }
        
        return direction_features