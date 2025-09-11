"""
Command-line interface for corridor segmentation.
"""

import argparse
import sys
from pathlib import Path
import logging

from .config import Config
from .main import CorridorSegmenter


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automatic extraction of high-voltage power transmission objects from UAV LiDAR point clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m corridor_seg --input data/power_corridor.las --outdir ./results
  python -m corridor_seg --input *.las --outdir ./batch_results --config config.yaml
  python -m corridor_seg --input scan.las --outdir ./out --visualize
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input LAS file path (or glob pattern for batch processing)')
    parser.add_argument('--outdir', '-o', type=str, required=True,
                       help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--config', '-c', type=str,
                       help='YAML configuration file path (overrides defaults)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Enable visualization of results')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate processing results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    # Processing parameters (override config)
    parser.add_argument('--grid-size', type=float,
                       help='2D grid size in meters (default: 5.0)')
    parser.add_argument('--voxel-size', type=float,
                       help='3D voxel size in meters (default: 0.5)')
    parser.add_argument('--min-height-gap', type=float,
                       help='Minimum height gap in meters (default: 8.0)')
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        if args.config:
            config = Config(args.config)
        else:
            config = Config()
        
        # Override config with command line arguments
        if args.visualize:
            config.enable_visualization = True
        if args.save_intermediate:
            config.save_intermediate = True
        if args.log_level:
            config.log_level = args.log_level
        if args.grid_size:
            config.grid_2d_size = args.grid_size
        if args.voxel_size:
            config.voxel_size = args.voxel_size
        if args.min_height_gap:
            config.min_height_gap = args.min_height_gap
        
        # Initialize segmenter
        segmenter = CorridorSegmenter(config)
        
        # Process input files
        input_path = Path(args.input)
        output_dir = Path(args.outdir)
        
        if '*' in str(input_path):
            # Batch processing
            print(f"Batch processing: {args.input}")
            las_files = list(Path('.').glob(args.input))
            
            if not las_files:
                print(f"No files found matching pattern: {args.input}")
                sys.exit(1)
            
            for las_file in las_files:
                print(f"\nProcessing: {las_file}")
                file_output_dir = output_dir / las_file.stem
                
                try:
                    results = segmenter.segment_corridor(str(las_file), str(file_output_dir))
                    print(f"✓ Successfully processed {las_file}")
                    print(f"  Power lines: {len(results['power_lines'])}")
                    print(f"  Towers: {len(results['towers'])}")
                except Exception as e:
                    print(f"✗ Error processing {las_file}: {e}")
                    continue
        
        else:
            # Single file processing
            if not input_path.exists():
                print(f"Input file not found: {args.input}")
                sys.exit(1)
            
            print(f"Processing: {args.input}")
            results = segmenter.segment_corridor(args.input, args.outdir)
            
            print(f"✓ Processing complete!")
            print(f"Results saved to: {args.outdir}")
            print(f"Power lines detected: {len(results['power_lines'])}")
            print(f"Towers detected: {len(results['towers'])}")
            
            # Performance summary
            stats = results['processing_stats']
            print(f"Processing time: {stats['total_time']:.2f}s")
            print(f"Points processed: {stats['point_counts']['filtered']:,}")
        
        # Optional visualization
        if config.enable_visualization and 'results' in locals():
            try:
                from .visualization import visualize_results
                print("\nOpening 3D visualization...")
                visualize_results(results)
            except ImportError:
                print("Visualization not available (missing dependencies)")
            except Exception as e:
                print(f"Visualization error: {e}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Detailed error information:")
        sys.exit(1)


if __name__ == '__main__':
    main()