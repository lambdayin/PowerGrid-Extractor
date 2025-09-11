"""
Configuration management for corridor segmentation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration class for power transmission object extraction."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with default values from paper."""
        
        # Core grid and voxel parameters (from paper)
        self.grid_2d_size = 5.0  # meters, 5m x 5m 2D grids
        self.voxel_size = 0.5    # meters, 0.5m x 0.5m x 0.5m voxels
        
        # Preprocessing parameters
        self.sor_k = 20          # Statistical outlier removal k neighbors
        self.sor_sigma = 2.0     # SOR sigma multiplier
        
        # Height analysis parameters
        self.min_height_gap = 8.0      # Δh_min: minimum height gap (meters)
        self.tower_head_height = None  # H_h: estimated from height histogram
        
        # Feature calculation parameters
        self.a1d_linear_thr = 0.6      # Linear dimensionality threshold
        self.a2d_planar_thr = 0.3      # Planar dimensionality threshold
        self.a3d_spherical_thr = 0.1   # Spherical dimensionality threshold
        
        # Power line extraction parameters
        self.collinearity_angle_thr = 10.0  # degrees, for PL segment merging
        self.neighbor_mode = "8-connected"   # 2D grid neighborhood mode
        
        # Tower segmentation parameters
        self.moving_window_size = 2    # 2x2 moving window for refinement
        self.tower_grid_cluster_offset = 5.0  # H_h + 5 meters for clustering
        self.planar_radius_offset = 5.0       # r + 5 for radius constraint
        
        # Topological optimization parameters
        self.connection_distance_thr = 10.0   # Max distance for PL-tower connection
        self.parallel_angle_thr = 15.0        # Angle threshold for parallel lines
        
        # Output parameters
        self.enable_visualization = True
        self.save_intermediate = False
        self.log_level = "INFO"
        
        # Load custom configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Update configuration with loaded values
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to YAML file."""
        config_data = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_data[key] = value
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = "Configuration Parameters:\n"
        config_str += "=" * 30 + "\n"
        
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"{key}: {value}\n"
        
        return config_str