#!/usr/bin/env python3
"""
Allocentric Synthetic Navigation Data Generator

Main entry point for generating synthetic navigation training data
using ALLOCENTRIC (absolute compass direction) action space.

Action Space: north, south, east, west, up, down, stop

Usage:
    python3 generate_allocentric_data.py \
        --scene_dir ../dataset/Matterport/1LXtFkjw3qL \
        --scene_name 1LXtFkjw3qL \
        --num_samples 100 \
        --output_dir ./allocentric_data \
        --step_size 1.0
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic navigation data with ALLOCENTRIC actions'
    )
    
    parser.add_argument(
        '--scene_dir',
        type=str,
        required=True,
        help='Path to scene directory containing CSV files'
    )
    
    parser.add_argument(
        '--scene_name',
        type=str,
        required=True,
        help='Scene name/ID'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to generate (default: 100)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./allocentric_data',
        help='Output directory (default: ./allocentric_data)'
    )
    
    parser.add_argument(
        '--step_size',
        type=float,
        default=1.0,
        help='Distance per action in meters (default: 1.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--show_examples',
        type=int,
        default=5,
        help='Number of example samples to print (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Construct paths to CSV files
    scene_dir = Path(args.scene_dir)
    region_csv = scene_dir / f"{args.scene_name}_region_result.csv"
    object_csv = scene_dir / f"{args.scene_name}_object_result.csv"
    
    # Validate input files
    if not region_csv.exists():
        print(f"ERROR: Region CSV not found: {region_csv}")
        sys.exit(1)
    
    if not object_csv.exists():
        print(f"ERROR: Object CSV not found: {object_csv}")
        sys.exit(1)
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        scene_name=args.scene_name,
        region_csv=str(region_csv),
        object_csv=str(object_csv),
        step_size=args.step_size,
        seed=args.seed
    )
    
    # Generate dataset
    samples = generator.generate_dataset(args.num_samples)
    
    if not samples:
        print("\nERROR: Failed to generate any samples")
        sys.exit(1)
    
    # Print statistics
    generator.print_statistics(samples)
    
    # Print examples
    if args.show_examples > 0:
        generator.print_examples(samples, args.show_examples)
    
    # Save dataset
    output_path = Path(args.output_dir) / f"{args.scene_name}_allocentric_data.json"
    generator.save_dataset(samples, str(output_path))
    
    print(f"\n{'='*80}")
    print("✓ Generation complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
