#!/usr/bin/env python3
"""
Compare multiple navigation samples side-by-side
"""

import argparse
import json
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path


def create_sample_visualization(scene_path, sample, step_size=1.0):
    """Create a plotter with a single sample"""
    scene_path = Path(scene_path)
    scene_name = scene_path.name
    
    # Load data
    pcd_path = scene_path / f'{scene_name}_pc_result.ply'
    region_csv = scene_path / f'{scene_name}_region_result.csv'
    
    region_df = pd.read_csv(region_csv)
    
    plotter = pv.Plotter(off_screen=False)
    plotter.set_background('white')
    
    # Add regions (very transparent)
    for _, row in region_df.iterrows():
        center = np.array([row['region_bbox_cx'], row['region_bbox_cy'], row['region_bbox_cz']])
        size = np.array([row['region_bbox_xlength'], row['region_bbox_ylength'], row['region_bbox_zlength']])
        box = pv.Cube(center=center, x_length=size[0], y_length=size[1], z_length=size[2])
        plotter.add_mesh(box, color='blue', opacity=0.05, style='wireframe')
    
    # Add trajectory
    coords = np.array(sample['path_coordinates'])
    if len(coords) >= 2:
        spline = pv.Spline(coords, len(coords) * 10)
        tube = spline.tube(radius=0.05)
        plotter.add_mesh(tube, color='yellow')
    
    # Add start/goal
    if len(coords) > 0:
        plotter.add_mesh(pv.Sphere(radius=0.2, center=coords[0]), color='green')
        plotter.add_mesh(pv.Sphere(radius=0.2, center=coords[-1]), color='red')
    
    # Add title
    plotter.add_text(
        f"{sample['task_id']}\n{sample['instruction'][:80]}...\n"
        f"Actions: {sample['num_actions']}, Length: {sample['path_length']:.1f}m",
        position='upper_left',
        font_size=8
    )
    
    return plotter


def main():
    parser = argparse.ArgumentParser(description='Compare navigation samples')
    parser.add_argument('--scene_path', type=str, required=True)
    parser.add_argument('--navigation_data', type=str, required=True)
    parser.add_argument('--samples', type=int, nargs='+', required=True,
                       help='Sample indices to compare (e.g., 0 5 10)')
    parser.add_argument('--step_size', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load navigation data
    with open(args.navigation_data, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # Validate indices
    for idx in args.samples:
        if idx < 0 or idx >= len(samples):
            print(f"Error: Sample index {idx} out of range (0-{len(samples)-1})")
            return 1
    
    # Create comparison view
    num_samples = len(args.samples)
    
    if num_samples == 2:
        shape = (1, 2)
    elif num_samples <= 4:
        shape = (2, 2)
    elif num_samples <= 6:
        shape = (2, 3)
    else:
        shape = (3, 3)
    
    plotter = pv.Plotter(shape=shape, window_size=(1920, 1080))
    
    for i, sample_idx in enumerate(args.samples):
        row = i // shape[1]
        col = i % shape[1]
        plotter.subplot(row, col)
        
        sample = samples[sample_idx]
        
        # Load data
        scene_path = Path(args.scene_path)
        scene_name = scene_path.name
        region_csv = scene_path / f'{scene_name}_region_result.csv'
        region_df = pd.read_csv(region_csv)
        
        # Add regions
        for _, r in region_df.iterrows():
            center = np.array([r['region_bbox_cx'], r['region_bbox_cy'], r['region_bbox_cz']])
            size = np.array([r['region_bbox_xlength'], r['region_bbox_ylength'], r['region_bbox_zlength']])
            box = pv.Cube(center=center, x_length=size[0], y_length=size[1], z_length=size[2])
            plotter.add_mesh(box, color='blue', opacity=0.05, style='wireframe')
        
        # Add trajectory
        coords = np.array(sample['path_coordinates'])
        if len(coords) >= 2:
            spline = pv.Spline(coords, len(coords) * 10)
            tube = spline.tube(radius=0.05)
            plotter.add_mesh(tube, color='yellow')
        
        # Add markers
        if len(coords) > 0:
            plotter.add_mesh(pv.Sphere(radius=0.2, center=coords[0]), color='green')
            plotter.add_mesh(pv.Sphere(radius=0.2, center=coords[-1]), color='red')
        
        # Add title
        plotter.add_text(
            f"Sample {sample_idx}: {sample['task_id']}\n"
            f"{sample['instruction'][:60]}...\n"
            f"Actions: {sample['num_actions']}, Length: {sample['path_length']:.1f}m",
            font_size=8
        )
        
        plotter.reset_camera()
    
    plotter.show()


if __name__ == '__main__':
    main()
