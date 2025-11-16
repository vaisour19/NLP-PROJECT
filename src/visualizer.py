"""
Visualization utilities for floor decomposition
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

try:
    from .floor_detector import FloorSlab
except ImportError:
    from floor_detector import FloorSlab


class FloorVisualizer:
    """Visualize floor decomposition results"""
    
    @staticmethod
    def plot_floor_distribution(floors: List[FloorSlab], scene_name: str, output_path: Path = None):
        """
        Plot floor Z-ranges and region distribution
        
        Args:
            floors: List of detected floors
            scene_name: Scene identifier
            output_path: Optional path to save figure
        """
        if not floors:
            print("No floors to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Floor Z-ranges
        floor_ids = [f.floor_id for f in floors]
        z_mins = [f.z_min for f in floors]
        z_maxs = [f.z_max for f in floors]
        z_centers = [f.z_center for f in floors]
        heights = [f.z_max - f.z_min for f in floors]
        
        for i, floor in enumerate(floors):
            ax1.barh(i, heights[i], left=z_mins[i], height=0.6, 
                    label=f'Floor {i}', alpha=0.7)
            ax1.text(z_centers[i], i, f'{heights[i]:.2f}m', 
                    ha='center', va='center', fontweight='bold')
        
        ax1.set_xlabel('Z Coordinate (meters)')
        ax1.set_ylabel('Floor ID')
        ax1.set_title(f'Floor Z-Ranges: {scene_name}')
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_yticks(floor_ids)
        
        # Plot 2: Regions per floor
        region_counts = [len(f.region_ids) for f in floors]
        
        bars = ax2.bar(floor_ids, region_counts, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Floor ID')
        ax2.set_ylabel('Number of Regions')
        ax2.set_title(f'Regions per Floor: {scene_name}')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, region_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def print_floor_summary(floors: List[FloorSlab], scene_name: str):
        """Print text summary of floor decomposition"""
        print(f"\n{'='*70}")
        print(f"Floor Decomposition Summary: {scene_name}")
        print(f"{'='*70}")
        print(f"Total floors detected: {len(floors)}")
        print(f"{'-'*70}")
        
        for floor in floors:
            height = floor.z_max - floor.z_min
            print(f"\nFloor {floor.floor_id}:")
            print(f"  Z-range: [{floor.z_min:.3f}, {floor.z_max:.3f}] (height: {height:.3f}m)")
            print(f"  Regions: {len(floor.region_ids)} regions")
            print(f"  Region IDs: {floor.region_ids[:10]}{'...' if len(floor.region_ids) > 10 else ''}")
        
        print(f"\n{'='*70}\n")
    
    @staticmethod
    def visualize_floor_files(floor_dir: Path):
        """Print summary of generated floor files"""
        floor_dir = Path(floor_dir)
        
        if not floor_dir.exists():
            print(f"Floor directory not found: {floor_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"Generated Files: {floor_dir.name}")
        print(f"{'='*70}")
        
        # List all files
        files = sorted(floor_dir.glob('*'))
        
        for file in files:
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name:<50} ({size_kb:>8.2f} KB)")
            
            # Show quick stats for CSV files
            if file.suffix == '.csv':
                df = pd.read_csv(file)
                print(f"    → {len(df)} rows, {len(df.columns)} columns")
            
            # Show quick stats for JSON files
            elif file.suffix == '.json':
                with open(file, 'r') as f:
                    data = json.load(f)
                if 'regions' in data:
                    print(f"    → {len(data['regions'])} regions")
        
        print(f"{'='*70}\n")
