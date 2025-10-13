#!/usr/bin/env python3
"""
Batch visualization script - generate images for all samples
"""

import argparse
import json
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
from tqdm import tqdm


def visualize_sample_to_image(scene_path, sample, output_path, step_size=1.0):
    """Generate visualization image for a single sample"""
    scene_path = Path(scene_path)
    scene_name = scene_path.name
    
    # Load data
    pcd_path = scene_path / f'{scene_name}_pc_result.ply'
    region_csv = scene_path / f'{scene_name}_region_result.csv'
    object_csv = scene_path / f'{scene_name}_object_result.csv'
    
    region_df = pd.read_csv(region_csv)
    object_df = pd.read_csv(object_csv)
    non_navigable = ['wall', 'ceiling', 'floor', 'void', 'remove']
    object_df = object_df[~object_df['nyu40_label'].isin(non_navigable)]
    
    # Create plotter (off-screen)
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background('white')
    
    # Add point cloud
    if pcd_path.exists():
        try:
            point_cloud = pv.read(str(pcd_path))
            plotter.add_mesh(point_cloud, opacity=0.3, point_size=2)
        except:
            pass
    
    # Add regions (wireframe)
    for _, row in region_df.iterrows():
        center = np.array([row['region_bbox_cx'], row['region_bbox_cy'], row['region_bbox_cz']])
        size = np.array([row['region_bbox_xlength'], row['region_bbox_ylength'], row['region_bbox_zlength']])
        box = pv.Cube(center=center, x_length=size[0], y_length=size[1], z_length=size[2])
        plotter.add_mesh(box, color='blue', opacity=0.1, style='wireframe', line_width=1)
    
    # Add trajectory
    coords = np.array(sample['path_coordinates'])
    if len(coords) >= 2:
        spline = pv.Spline(coords, len(coords) * 10)
        tube = spline.tube(radius=0.05)
        plotter.add_mesh(tube, color='yellow', opacity=1.0)
    
    # Add start marker
    if len(coords) > 0:
        sphere = pv.Sphere(radius=0.25, center=coords[0])
        plotter.add_mesh(sphere, color='green', opacity=0.9)
        plotter.add_point_labels([coords[0]], ['START'], font_size=20, point_color='green')
    
    # Add goal marker
    if len(coords) > 0:
        sphere = pv.Sphere(radius=0.25, center=coords[-1])
        plotter.add_mesh(sphere, color='red', opacity=0.9)
        plotter.add_point_labels([coords[-1]], ['GOAL'], font_size=20, point_color='red')
    
    # Add action arrows
    action_vectors = {
        'north': np.array([0, 1, 0]), 'south': np.array([0, -1, 0]),
        'east': np.array([1, 0, 0]), 'west': np.array([-1, 0, 0]),
        'up': np.array([0, 0, 1]), 'down': np.array([0, 0, -1])
    }
    
    action_colors = {
        'north': '#FF0000', 'south': '#0000FF', 'east': '#00FF00',
        'west': '#FFFF00', 'up': '#FF00FF', 'down': '#00FFFF'
    }
    
    current_pos = coords[0].copy()
    arrow_spacing = 0
    
    for action in sample['action_sequence']:
        if action == 'stop' or action not in action_vectors:
            break
        
        direction = action_vectors[action]
        arrow_spacing += 1
        
        if arrow_spacing % 3 == 0:
            arrow = pv.Arrow(start=current_pos, direction=direction, scale=step_size * 0.8,
                           tip_length=0.3, tip_radius=0.15, shaft_radius=0.05)
            plotter.add_mesh(arrow, color=action_colors.get(action, '#888888'), opacity=0.8)
        
        current_pos += direction * step_size
    
    # Add title
    plotter.add_text(
        f"Sample {sample['task_id']}\n{sample['instruction'][:100]}...",
        position='upper_left',
        font_size=10,
        color='black'
    )
    
    # Reset camera and save
    plotter.reset_camera()
    plotter.screenshot(output_path)
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description='Batch generate visualization images')
    parser.add_argument('--scene_path', type=str, required=True)
    parser.add_argument('--navigation_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./visualizations')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--step_size', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load navigation data
    with open(args.navigation_data, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(samples)} visualization images...")
    
    # Generate images
    for i, sample in enumerate(tqdm(samples)):
        output_path = output_dir / f"sample_{i:04d}.png"
        try:
            visualize_sample_to_image(args.scene_path, sample, output_path, args.step_size)
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
    
    print(f"\n✓ Done! Images saved to {output_dir}")


if __name__ == '__main__':
    main()
