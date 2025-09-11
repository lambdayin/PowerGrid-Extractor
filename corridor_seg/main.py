"""
Main corridor segmentation module.
Orchestrates the complete power transmission object extraction pipeline.
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import laspy

from .config import Config
from .preprocessing import PointCloudPreprocessor
from .features import FeatureCalculator
from .powerlines import PowerLineExtractor
from .towers import TowerExtractor
from .topology import TopologyOptimizer


class CorridorSegmenter:
    """Main class for power transmission object extraction."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize corridor segmenter with configuration."""
        self.config = config or Config()
        self._setup_logging()
        
        # Initialize processing modules
        self.preprocessor = PointCloudPreprocessor(self.config)
        self.feature_calculator = FeatureCalculator(self.config)
        self.powerline_extractor = PowerLineExtractor(self.config)
        self.tower_extractor = TowerExtractor(self.config)
        self.topology_optimizer = TopologyOptimizer(self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('corridor_segmentation.log')
            ]
        )
    
    def load_point_cloud(self, input_path: str) -> np.ndarray:
        """
        Load point cloud from LAS file.
        
        Args:
            input_path: Path to input LAS file
            
        Returns:
            points: Point cloud array (N, 3)
        """
        self.logger.info(f"Loading point cloud from: {input_path}")
        
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_file.suffix.lower() != '.las':
            raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
        # Load LAS file
        las_file = laspy.read(input_path)
        
        # Extract XYZ coordinates
        points = np.vstack([las_file.x, las_file.y, las_file.z]).T
        
        self.logger.info(f"Loaded {len(points)} points from {input_path}")
        self.logger.info(f"Point cloud bounds: "
                        f"X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                        f"Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                        f"Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        return points
    
    def save_point_cloud(self, points: np.ndarray, output_path: str, 
                        original_las_path: Optional[str] = None) -> None:
        """
        Save point cloud to LAS file.
        
        Args:
            points: Point cloud array (N, 3)
            output_path: Output LAS file path
            original_las_path: Original LAS file for header copying
        """
        self.logger.info(f"Saving {len(points)} points to: {output_path}")
        
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create new LAS file
        if original_las_path and Path(original_las_path).exists():
            # Copy header from original file
            original_las = laspy.read(original_las_path)
            header = original_las.header
        else:
            # Create minimal header with compatible version
            header = laspy.LasHeader(point_format=0, version=(1, 4))
        
        # Create new LAS file with default compatible format
        las_out = laspy.create()
        
        las_out.x = points[:, 0]
        las_out.y = points[:, 1] 
        las_out.z = points[:, 2]
        
        # Set default values for required fields
        if hasattr(las_out, 'intensity'):
            las_out.intensity = np.zeros(len(points), dtype=np.uint16)
        if hasattr(las_out, 'return_number'):
            las_out.return_number = np.ones(len(points), dtype=np.uint8)
        if hasattr(las_out, 'number_of_returns'):
            las_out.number_of_returns = np.ones(len(points), dtype=np.uint8)
        
        # Write to file
        las_out.write(output_path)
        
        self.logger.info(f"Successfully saved point cloud to: {output_path}")
    
    def process_point_cloud(self, points: np.ndarray) -> Dict:
        """
        Complete point cloud processing pipeline.
        
        Args:
            points: Input point cloud
            
        Returns:
            results: Complete processing results
        """
        start_time = time.time()
        self.logger.info("Starting complete corridor segmentation pipeline")
        
        # Stage 1: Preprocessing
        self.logger.info("=== Stage 1: Preprocessing ===")
        stage1_start = time.time()
        preprocessed_data = self.preprocessor.preprocess(points)
        filtered_points = preprocessed_data['points']
        grid_2d = preprocessed_data['grid_2d']
        voxel_hash_3d = preprocessed_data['voxel_hash_3d']
        delta_h_min = preprocessed_data['delta_h_min']
        tower_head_height = preprocessed_data['tower_head_height']
        stage1_time = time.time() - stage1_start
        self.logger.info(f"Stage 1 completed in {stage1_time:.2f}s")
        
        # Stage 2: Feature Calculation
        self.logger.info("=== Stage 2: Feature Calculation ===")
        stage2_start = time.time()
        
        # Compute voxel features (3D dimensional features)
        voxel_features = self.feature_calculator.compute_voxel_features(
            voxel_hash_3d, filtered_points)
        
        # Identify linear structures
        linear_voxels = self.feature_calculator.identify_linear_structures(voxel_features)
        
        # Initial 2D grid features (without PL exclusion)
        grid_features_initial = self.feature_calculator.compute_2d_grid_features(
            grid_2d, filtered_points, pl_candidate_mask=None)
        
        stage2_time = time.time() - stage2_start
        self.logger.info(f"Stage 2 completed in {stage2_time:.2f}s")
        
        # Stage 3: Power Line Extraction
        self.logger.info("=== Stage 3: Power Line Extraction ===")
        stage3_start = time.time()
        
        # Compute grid origin from preprocessing bounds for consistent indexing
        bounds = preprocessed_data.get('bounds', {})
        min_coords = bounds.get('min_coords', None)
        grid_origin = None
        if min_coords is not None and len(min_coords) >= 2:
            grid_origin = (float(min_coords[0]), float(min_coords[1]))
        
        power_lines, pl_mask = self.powerline_extractor.extract_power_lines(
            linear_voxels, voxel_features, filtered_points, grid_2d, delta_h_min,
            grid_origin=grid_origin)
        
        stage3_time = time.time() - stage3_start
        self.logger.info(f"Stage 3 completed in {stage3_time:.2f}s")
        
        # Stage 4: Recompute Grid Features (AFTER PL extraction)
        self.logger.info("=== Stage 4: Grid Features Refinement ===")
        stage4_start = time.time()
        
        # CRITICAL: Recompute DSM/nDSM features after removing PL points
        grid_features = self.feature_calculator.compute_2d_grid_features(
            grid_2d, filtered_points, pl_candidate_mask=pl_mask)
        
        # Enhance with height-based features
        grid_features = self.feature_calculator.compute_height_based_features(
            grid_features, delta_h_min, tower_head_height)
        
        stage4_time = time.time() - stage4_start
        self.logger.info(f"Stage 4 completed in {stage4_time:.2f}s")
        
        # Stage 5: Tower Extraction
        self.logger.info("=== Stage 5: Tower Extraction ===")
        stage5_start = time.time()
        
        towers, tower_mask = self.tower_extractor.extract_tower_candidates(
            grid_features, filtered_points, delta_h_min, tower_head_height,
            grid_2d=grid_2d, grid_origin=grid_origin)
        
        stage5_time = time.time() - stage5_start
        self.logger.info(f"Stage 5 completed in {stage5_time:.2f}s")
        
        # Stage 6: Topological Optimization
        self.logger.info("=== Stage 6: Topological Optimization ===")
        stage6_start = time.time()
        
        optimized_power_lines, optimized_towers, topology_report = \
            self.topology_optimizer.optimize_topology(power_lines, towers)
        
        # Update masks with optimized results
        final_pl_mask = np.zeros(len(filtered_points), dtype=bool)
        final_tower_mask = np.zeros(len(filtered_points), dtype=bool)
        
        for pl in optimized_power_lines:
            if 'point_indices' in pl:
                final_pl_mask[pl['point_indices']] = True
        
        for tower in optimized_towers:
            if 'points' in tower and len(tower['points']) > 0:
                # Find indices in filtered_points
                tower_points = tower['points']
                for tp in tower_points:
                    distances = np.sum((filtered_points - tp)**2, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 0.01:  # 1cm tolerance
                        final_tower_mask[closest_idx] = True
        
        stage6_time = time.time() - stage6_start
        self.logger.info(f"Stage 6 completed in {stage6_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'points_original': points,
            'points_filtered': filtered_points,
            'preprocessing_data': preprocessed_data,
            'voxel_features': voxel_features,
            'grid_features': grid_features,
            'linear_voxels': linear_voxels,
            'power_lines': optimized_power_lines,
            'towers': optimized_towers,
            'pl_mask': final_pl_mask,
            'tower_mask': final_tower_mask,
            'pl_points': filtered_points[final_pl_mask],
            'tower_points': filtered_points[final_tower_mask],
            'topology_report': topology_report,
            'processing_stats': {
                'total_time': total_time,
                'stage_times': {
                    'preprocessing': stage1_time,
                    'feature_calculation': stage2_time,
                    'powerline_extraction': stage3_time,
                    'grid_refinement': stage4_time,
                    'tower_extraction': stage5_time,
                    'topology_optimization': stage6_time
                },
                'point_counts': {
                    'original': len(points),
                    'filtered': len(filtered_points),
                    'powerlines': final_pl_mask.sum(),
                    'towers': final_tower_mask.sum()
                }
            }
        }
        
        self.logger.info(f"Complete pipeline finished in {total_time:.2f}s")
        self.logger.info(f"Results: {len(optimized_power_lines)} power lines, "
                        f"{len(optimized_towers)} towers")
        self.logger.info(f"Point distribution: {final_pl_mask.sum()} PL points, "
                        f"{final_tower_mask.sum()} tower points")
        
        return results
    
    def segment_corridor(self, input_path: str, output_dir: str) -> Dict:
        """
        Complete corridor segmentation from LAS file input to LAS file outputs.
        
        Args:
            input_path: Input LAS file path
            output_dir: Output directory for results
            
        Returns:
            results: Processing results
        """
        self.logger.info(f"Starting corridor segmentation: {input_path} -> {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load point cloud
        points = self.load_point_cloud(input_path)
        
        # Process point cloud
        results = self.process_point_cloud(points)
        
        # Save results
        self.logger.info("Saving segmentation results")
        
        # Save power line points
        pl_points = results['pl_points']
        if len(pl_points) > 0:
            pl_output_path = output_path / 'powerlines.las'
            self.save_point_cloud(pl_points, str(pl_output_path), input_path)
        else:
            self.logger.warning("No power line points to save")
        
        # Save tower points
        tower_points = results['tower_points'] 
        if len(tower_points) > 0:
            tower_output_path = output_path / 'towers.las'
            self.save_point_cloud(tower_points, str(tower_output_path), input_path)
        else:
            self.logger.warning("No tower points to save")
        
        # Save processing log
        self._save_processing_report(results, output_path / 'processing_report.txt')
        
        # Optional: Save intermediate results
        if self.config.save_intermediate:
            self._save_intermediate_results(results, output_path)
        
        self.logger.info(f"Corridor segmentation complete. Results saved to: {output_dir}")
        
        return results
    
    def _save_processing_report(self, results: Dict, report_path: Path) -> None:
        """Save detailed processing report."""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Power Transmission Objects Extraction Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Processing statistics
            stats = results['processing_stats']
            f.write("Processing Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total processing time: {stats['total_time']:.2f}s\n")
            f.write(f"Original points: {stats['point_counts']['original']:,}\n")
            f.write(f"Filtered points: {stats['point_counts']['filtered']:,}\n")
            f.write(f"Power line points: {stats['point_counts']['powerlines']:,}\n")
            f.write(f"Tower points: {stats['point_counts']['towers']:,}\n\n")
            
            # Stage timing
            f.write("Stage Processing Times:\n")
            f.write("-" * 20 + "\n")
            for stage, time_val in stats['stage_times'].items():
                f.write(f"{stage}: {time_val:.2f}s\n")
            f.write("\n")
            
            # Detection results
            f.write("Detection Results:\n") 
            f.write("-" * 16 + "\n")
            f.write(f"Power lines detected: {len(results['power_lines'])}\n")
            f.write(f"Towers detected: {len(results['towers'])}\n\n")
            
            # Power line details
            if results['power_lines']:
                f.write("Power Line Details:\n")
                f.write("-" * 18 + "\n")
                for pl in results['power_lines']:
                    f.write(f"PL {pl['powerline_id']}: "
                           f"Length={pl.get('total_length', 0):.1f}m, "
                           f"Points={len(pl.get('point_indices', []))}, "
                           f"Height={pl.get('height_stats', {}).get('mean', 0):.1f}m\n")
                f.write("\n")
            
            # Tower details
            if results['towers']:
                f.write("Tower Details:\n")
                f.write("-" * 13 + "\n")
                for tower in results['towers']:
                    f.write(f"Tower {tower['cluster_id']}: "
                           f"Height={tower.get('max_height_diff', 0):.1f}m, "
                           f"Points={tower.get('total_points', 0)}, "
                           f"Cells={tower.get('num_cells', 0)}\n")
                f.write("\n")
            
            # Topology report
            if 'topology_report' in results:
                topo = results['topology_report']
                f.write("Topological Analysis:\n")
                f.write("-" * 19 + "\n")
                f.write(f"PL-Tower connections: {topo['connection_analysis'].get('valid_connections', 0)}\n")
                f.write(f"Parallel violations: {len(topo['parallel_analysis'].get('violations', []))}\n")
                f.write(f"Extremal points: {topo['extremal_analysis'].get('num_extrema', 0)}\n")
                f.write(f"Optimization success: {topo.get('optimization_success', False)}\n")
    
    def _save_intermediate_results(self, results: Dict, output_path: Path) -> None:
        """Save intermediate processing results."""
        intermediate_dir = output_path / 'intermediate'
        intermediate_dir.mkdir(exist_ok=True)
        
        # Save filtered points
        filtered_points = results['points_filtered']
        if len(filtered_points) > 0:
            filtered_path = intermediate_dir / 'filtered_points.las'
            self.save_point_cloud(filtered_points, str(filtered_path))
        
        # Additional intermediate saves can be added here
        self.logger.info(f"Intermediate results saved to: {intermediate_dir}")