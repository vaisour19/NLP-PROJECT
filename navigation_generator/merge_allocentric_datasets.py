#!/usr/bin/env python3
"""
Merge multiple allocentric navigation datasets into one file
and optionally split into train/val/test sets
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np


def load_datasets(input_dir: str) -> List[Dict]:
    """Load all JSON datasets from directory"""
    all_samples = []
    
    json_files = sorted(Path(input_dir).glob('*_allocentric_data.json'))
    
    if not json_files:
        print(f"No allocentric data files found in {input_dir}")
        return []
    
    print(f"Found {len(json_files)} dataset files")
    print("-" * 80)
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        
        samples = data['samples']
        scene_name = data['scene_name']
        
        print(f"  {scene_name}: {len(samples)} samples")
        all_samples.extend(samples)
    
    print("-" * 80)
    print(f"Total samples loaded: {len(all_samples)}")
    
    return all_samples


def split_dataset(samples: List[Dict], 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> Dict[str, List[Dict]]:
    """Split dataset into train/val/test"""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    random.shuffle(samples)
    
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }
    
    print("\nDataset split:")
    print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/total*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/total*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/total*100:.1f}%)")
    
    return splits


def analyze_dataset(samples: List[Dict]):
    """Print dataset statistics"""
    
    if not samples:
        print("No samples to analyze")
        return
    
    print("\n" + "="*80)
    print("Dataset Analysis")
    print("="*80 + "\n")
    
    # Count unique scenes
    scenes = set(s['scene_name'] for s in samples)
    print(f"Unique scenes: {len(scenes)}")
    print(f"Total samples: {len(samples)}")
    
    # Vertical movement
    vertical_samples = [s for s in samples if s['has_vertical_movement']]
    print(f"Samples with vertical movement: {len(vertical_samples)} "
          f"({len(vertical_samples)/len(samples)*100:.1f}%)")
    
    # Average statistics
    avg_actions = np.mean([s['num_actions'] for s in samples])
    avg_path_length = np.mean([s['path_length'] for s in samples])
    
    print(f"\nAverage statistics:")
    print(f"  Actions per sample: {avg_actions:.1f}")
    print(f"  Path length: {avg_path_length:.2f}m")
    
    if vertical_samples:
        avg_vertical = np.mean([s['vertical_distance'] for s in vertical_samples])
        print(f"  Vertical distance (when present): {avg_vertical:.2f}m")
    
    # Action distribution
    all_actions = []
    for sample in samples:
        all_actions.extend(sample['action_sequence'])
    
    action_counts = {}
    for action in all_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    total_actions = len(all_actions)
    print(f"\nAction distribution:")
    for action in ['north', 'south', 'east', 'west', 'up', 'down', 'stop']:
        count = action_counts.get(action, 0)
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"  {action:10s}: {count:6d} ({percentage:5.1f}%)")
    
    print("\n" + "="*80)


def save_dataset(samples: List[Dict], output_path: str, split_name: str = None):
    """Save dataset to JSON file"""
    
    output_data = {
        'num_samples': len(samples),
        'action_space': 'allocentric',
        'actions': ['north', 'south', 'east', 'west', 'up', 'down', 'stop'],
        'coordinate_system': {
            'x_axis': 'East (positive) / West (negative)',
            'y_axis': 'North (positive) / South (negative)',
            'z_axis': 'Up (positive) / Down (negative)'
        },
        'samples': samples
    }
    
    if split_name:
        output_data['split'] = split_name
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    size_kb = output_path.stat().st_size / 1024
    print(f"\n✓ Saved: {output_path} ({size_kb:.2f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description='Merge allocentric navigation datasets'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing JSON dataset files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='merged_allocentric_dataset.json',
        help='Output file path (default: merged_allocentric_dataset.json)'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split into train/val/test sets'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Train set ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Allocentric Dataset Merger")
    print("="*80 + "\n")
    
    # Load all datasets
    samples = load_datasets(args.input_dir)
    
    if not samples:
        print("\nERROR: No samples loaded")
        return
    
    # Analyze combined dataset
    analyze_dataset(samples)
    
    if args.split:
        # Split dataset
        splits = split_dataset(
            samples,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
        
        # Save splits
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
        
        for split_name, split_samples in splits.items():
            output_path = output_dir / f"{output_base}_{split_name}.json"
            save_dataset(split_samples, str(output_path), split_name)
    else:
        # Save merged dataset
        save_dataset(samples, args.output)
    
    print("\n" + "="*80)
    print("✓ Merge complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
