"""
Unit tests for feature calculation module.
"""

import numpy as np
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corridor_seg.features import FeatureCalculator
from corridor_seg.config import Config


class TestFeatureCalculator(unittest.TestCase):
    """Test cases for FeatureCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.calculator = FeatureCalculator(self.config)
    
    def test_covariance_features_linear_structure(self):
        """Test dimensional features for linear structure (line)."""
        # Create a perfect line along X-axis
        t = np.linspace(0, 10, 100)
        points = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
        
        features = self.calculator.compute_covariance_features(points)
        
        # For a perfect line: a1D should be high, a2D and a3D should be low
        self.assertGreater(features['a1D'], 0.9, "Linear structure should have high a1D")
        self.assertLess(features['a2D'], 0.1, "Linear structure should have low a2D")
        self.assertLess(features['a3D'], 0.1, "Linear structure should have low a3D")
    
    def test_covariance_features_planar_structure(self):
        """Test dimensional features for planar structure."""
        # Create a plane in XY
        x, y = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
        points = np.column_stack([x.flatten(), y.flatten(), np.zeros(100)])
        
        features = self.calculator.compute_covariance_features(points)
        
        # For a plane: a2D should be high, a3D should be low
        self.assertGreater(features['a2D'], 0.5, "Planar structure should have high a2D")
        self.assertLess(features['a3D'], 0.1, "Planar structure should have low a3D")
    
    def test_covariance_features_spherical_structure(self):
        """Test dimensional features for spherical structure."""
        # Create a sphere
        np.random.seed(42)
        phi = np.random.uniform(0, 2*np.pi, 1000)
        costheta = np.random.uniform(-1, 1, 1000)
        theta = np.arccos(costheta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        points = np.column_stack([x, y, z])
        
        features = self.calculator.compute_covariance_features(points)
        
        # For a sphere: a3D should be relatively high
        self.assertGreater(features['a3D'], 0.3, "Spherical structure should have higher a3D")
    
    def test_covariance_features_empty_points(self):
        """Test behavior with insufficient points."""
        # Empty points
        points = np.empty((0, 3))
        features = self.calculator.compute_covariance_features(points)
        
        self.assertEqual(features['a1D'], 0.0)
        self.assertEqual(features['a2D'], 0.0)
        self.assertEqual(features['a3D'], 0.0)
        
        # Single point
        points = np.array([[1, 2, 3]])
        features = self.calculator.compute_covariance_features(points)
        
        self.assertEqual(features['a1D'], 0.0)
        self.assertEqual(features['a2D'], 0.0)
        self.assertEqual(features['a3D'], 0.0)
    
    def test_linear_structure_identification(self):
        """Test identification of linear structures."""
        # Create mock voxel features
        voxel_features = {
            (0, 0, 0): {'a1D': 0.8, 'a2D': 0.1, 'a3D': 0.05},  # Linear
            (1, 0, 0): {'a1D': 0.2, 'a2D': 0.7, 'a3D': 0.1},   # Planar
            (2, 0, 0): {'a1D': 0.3, 'a2D': 0.3, 'a3D': 0.4},   # Spherical
        }
        
        linear_voxels = self.calculator.identify_linear_structures(voxel_features)
        
        # Only the first voxel should be identified as linear
        self.assertEqual(len(linear_voxels), 1)
        self.assertIn((0, 0, 0), linear_voxels)
        self.assertNotIn((1, 0, 0), linear_voxels)
        self.assertNotIn((2, 0, 0), linear_voxels)


class TestGridFeatures(unittest.TestCase):
    """Test cases for 2D grid feature calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.calculator = FeatureCalculator(self.config)
    
    def test_grid_features_basic(self):
        """Test basic 2D grid feature calculation."""
        # Create simple grid with points
        grid_2d = {
            (0, 0): [0, 1, 2],  # 3 points in cell (0,0)
            (1, 0): [3, 4],     # 2 points in cell (1,0)
        }
        
        points = np.array([
            [0, 0, 5],   # Ground level
            [1, 1, 10],  # Mid level  
            [2, 2, 15],  # High level
            [5, 0, 8],   # Different cell
            [6, 1, 12],  # Different cell
        ])
        
        grid_features = self.calculator.compute_2d_grid_features(grid_2d, points)
        
        # Check first cell features
        cell_features = grid_features[(0, 0)]
        self.assertEqual(cell_features['point_count'], 3)
        self.assertEqual(cell_features['DEM'], 5.0)  # Minimum height
        self.assertEqual(cell_features['DSM'], 15.0)  # Maximum height
        self.assertEqual(cell_features['HeightDiff'], 10.0)  # DSM - DEM
    
    def test_grid_features_pl_exclusion(self):
        """Test DSM calculation after PL point exclusion."""
        grid_2d = {(0, 0): [0, 1, 2]}
        points = np.array([
            [0, 0, 5],   # Ground
            [1, 1, 20],  # Tower (should contribute to DSM)
            [2, 2, 12],  # Power line (should be excluded)
        ])
        
        # Mask indicating power line points
        pl_mask = np.array([False, False, True])  # Third point is PL
        
        grid_features = self.calculator.compute_2d_grid_features(
            grid_2d, points, pl_candidate_mask=pl_mask)
        
        cell_features = grid_features[(0, 0)]
        
        # DSM should exclude PL point (height=12), so max should be 20
        self.assertEqual(cell_features['DEM'], 5.0)
        self.assertEqual(cell_features['DSM'], 20.0)  # Without PL point
        self.assertEqual(cell_features['HeightDiff'], 15.0)


if __name__ == '__main__':
    unittest.main()