"""
Main pipeline script for floor-based dataset decomposition
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from .floor_detector import FloorDetector
    from .dataset_splitter import DatasetSplitter
    from .visualizer import FloorVisualizer
except ImportError:
    from floor_detector import FloorDetector
    from dataset_splitter import DatasetSplitter
    from visualizer import FloorVisualizer


class DecompositionPipeline:
    """Complete pipeline for decomposing scenes into floors"""
    
    def __init__(self, dataset_root: Path, output_root: Path, 
                 z_tolerance: float = 0.4, exclude_stairs: bool = True,
                 visualize: bool = True):
        """
        Args:
            dataset_root: Root directory of VLA-3D dataset (e.g., VLA-3D_dataset/Matterport)
            output_root: Output directory for floor-split data
            z_tolerance: Tolerance for floor clustering (meters)
            exclude_stairs: Whether to exclude stairs from floor assignment
            visualize: Whether to generate visualizations
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.z_tolerance = z_tolerance
        self.exclude_stairs = exclude_stairs
        self.visualize = visualize
        
        self.detector = FloorDetector(z_tolerance=z_tolerance, exclude_stairs=exclude_stairs)
        self.visualizer = FloorVisualizer()
    
    def process_scene(self, scene_name: str) -> dict:
        """
        Process a single scene
        
        Returns:
            Dictionary with processing results
        """
        scene_dir = self.dataset_root / scene_name
        
        if not scene_dir.exists():
            return {
                'scene_name': scene_name,
                'success': False,
                'error': f'Scene directory not found: {scene_dir}'
            }
        
        region_csv = scene_dir / f"{scene_name}_region_result.csv"
        if not region_csv.exists():
            return {
                'scene_name': scene_name,
                'success': False,
                'error': f'Region CSV not found: {region_csv}'
            }
        
        try:
            # Step 1: Detect floors
            print(f"\n[{scene_name}] Detecting floors...")
            floors = self.detector.detect_floors(str(region_csv))
            
            if not floors:
                return {
                    'scene_name': scene_name,
                    'success': False,
                    'error': 'No floors detected'
                }
            
            # Validate floors
            validation = self.detector.validate_floors(floors)
            print(f"[{scene_name}] Detected {validation['num_floors']} floors")
            print(f"[{scene_name}] Total regions: {validation['total_regions']}")
            
            # Step 2: Visualize (optional)
            if self.visualize:
                self.visualizer.print_floor_summary(floors, scene_name)
                
                viz_dir = self.output_root / scene_name / 'visualizations'
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_path = viz_dir / f"{scene_name}_floor_distribution.png"
                self.visualizer.plot_floor_distribution(floors, scene_name, viz_path)
            
            # Step 3: Split dataset
            print(f"[{scene_name}] Splitting dataset into floors...")
            splitter = DatasetSplitter(scene_name, scene_dir, self.output_root)
            
            floor_dirs = []
            for floor in tqdm(floors, desc=f"Processing floors", leave=False):
                floor_dir = splitter.split_floor(floor)
                floor_dirs.append(floor_dir)
                
                if self.visualize:
                    self.visualizer.visualize_floor_files(floor_dir)
            
            print(f"[{scene_name}] ✓ Successfully processed {len(floors)} floors")
            
            return {
                'scene_name': scene_name,
                'success': True,
                'num_floors': len(floors),
                'floor_dirs': [str(d) for d in floor_dirs],
                'validation': validation
            }
            
        except Exception as e:
            return {
                'scene_name': scene_name,
                'success': False,
                'error': str(e)
            }
    
    def process_scenes(self, scene_names: list) -> dict:
        """
        Process multiple scenes
        
        Returns:
            Summary dictionary
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"Floor-Based Dataset Decomposition Pipeline")
        print(f"{'='*80}")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Output root: {self.output_root}")
        print(f"Z tolerance: {self.z_tolerance}m")
        print(f"Exclude stairs: {self.exclude_stairs}")
        print(f"Scenes to process: {len(scene_names)}")
        print(f"{'='*80}\n")
        
        for scene_name in scene_names:
            result = self.process_scene(scene_name)
            results.append(result)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"Processing Summary")
        print(f"{'='*80}")
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"Total scenes: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            total_floors = sum(r['num_floors'] for r in successful)
            print(f"\nTotal floors generated: {total_floors}")
            print(f"Average floors per scene: {total_floors / len(successful):.2f}")
        
        if failed:
            print(f"\n❌ Failed scenes:")
            for r in failed:
                print(f"  - {r['scene_name']}: {r['error']}")
        
        print(f"{'='*80}\n")
        
        return {
            'results': results,
            'num_successful': len(successful),
            'num_failed': len(failed),
            'total_floors': sum(r.get('num_floors', 0) for r in successful)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Decompose multi-floor VLA-3D scenes into single-floor sub-scenes'
    )
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to VLA-3D dataset root (e.g., VLA-3D_dataset/Matterport)')
    parser.add_argument('--output_root', type=str, required=True,
                       help='Output directory for floor-split data')
    parser.add_argument('--scenes', type=str, nargs='+', required=True,
                       help='Scene names to process')
    parser.add_argument('--z_tolerance', type=float, default=0.4,
                       help='Z-tolerance for floor clustering (meters)')
    parser.add_argument('--include_stairs', action='store_true',
                       help='Include stairs in floor assignment (default: exclude)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DecompositionPipeline(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        z_tolerance=args.z_tolerance,
        exclude_stairs=not args.include_stairs,
        visualize=not args.no_visualize
    )
    
    # Process scenes
    summary = pipeline.process_scenes(args.scenes)
    
    # Exit with appropriate code
    sys.exit(0 if summary['num_failed'] == 0 else 1)


if __name__ == '__main__':
    main()
