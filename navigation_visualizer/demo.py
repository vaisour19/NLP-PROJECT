#!/usr/bin/env python3
"""
Simple example script showing basic PyVista usage for navigation visualization
"""

import numpy as np
import pyvista as pv
import json
from pathlib import Path


def simple_visualizer_demo():
    """Minimal example of visualization"""
    
    # Example: Create a simple scene
    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.set_background('white')
    
    # Add a simple trajectory
    path_coords = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [2, 3, 0],
        [2, 3, 2],
        [5, 3, 2]
    ])
    
    # Create spline path
    spline = pv.Spline(path_coords, 50)
    tube = spline.tube(radius=0.05)
    plotter.add_mesh(tube, color='yellow', label='Trajectory')
    
    # Add start marker
    start_sphere = pv.Sphere(radius=0.2, center=path_coords[0])
    plotter.add_mesh(start_sphere, color='green', label='Start')
    
    # Add goal marker
    goal_sphere = pv.Sphere(radius=0.2, center=path_coords[-1])
    plotter.add_mesh(goal_sphere, color='red', label='Goal')
    
    # Add action arrows
    action_colors = {
        'east': 'green',
        'north': 'red',
        'up': 'magenta'
    }
    
    # East arrow
    arrow1 = pv.Arrow(start=[0, 0, 0], direction=[1, 0, 0], scale=2)
    plotter.add_mesh(arrow1, color=action_colors['east'], opacity=0.8)
    
    # North arrow
    arrow2 = pv.Arrow(start=[2, 0, 0], direction=[0, 1, 0], scale=3)
    plotter.add_mesh(arrow2, color=action_colors['north'], opacity=0.8)
    
    # Up arrow
    arrow3 = pv.Arrow(start=[2, 3, 0], direction=[0, 0, 1], scale=2)
    plotter.add_mesh(arrow3, color=action_colors['up'], opacity=0.8)
    
    # Add a room box (transparent)
    room_box = pv.Cube(center=[2.5, 1.5, 1], x_length=6, y_length=4, z_length=3)
    plotter.add_mesh(room_box, color='blue', opacity=0.1, style='wireframe', line_width=2)
    
    # Add axes
    plotter.add_axes(xlabel='X (East)', ylabel='Y (North)', zlabel='Z (Up)')
    
    # Add title
    plotter.add_text(
        'Simple Navigation Example\nGo east, then north, then up',
        position='upper_left',
        font_size=12,
        color='black'
    )
    
    # Show
    plotter.show()


def load_and_visualize_sample(scene_path: str, nav_data_path: str, sample_idx: int = 0):
    """Load and visualize a real sample (simplified version)"""
    
    # Load navigation data
    with open(nav_data_path, 'r') as f:
        data = json.load(f)
    
    sample = data['samples'][sample_idx]
    coords = np.array(sample['path_coordinates'])
    
    # Create plotter
    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.set_background('white')
    
    # Add trajectory
    if len(coords) >= 2:
        spline = pv.Spline(coords, len(coords) * 10)
        tube = spline.tube(radius=0.05)
        plotter.add_mesh(tube, color='yellow')
    
    # Add start/goal
    plotter.add_mesh(pv.Sphere(radius=0.25, center=coords[0]), color='green')
    plotter.add_mesh(pv.Sphere(radius=0.25, center=coords[-1]), color='red')
    
    # Add title
    plotter.add_text(
        f"Sample {sample_idx}\n{sample['instruction']}\n"
        f"Actions: {sample['num_actions']}, Length: {sample['path_length']:.2f}m",
        position='upper_left',
        font_size=10
    )
    
    plotter.add_axes()
    plotter.show()


if __name__ == '__main__':
    import sys
    
    print("PyVista Navigation Visualization Demo")
    print("=" * 60)
    print()
    print("1. Simple demo (synthetic trajectory)")
    print("2. Load and visualize real sample")
    print()
    
    choice = input("Select option (1 or 2): ").strip()
    
    if choice == '1':
        print("\nShowing simple demo...")
        simple_visualizer_demo()
    
    elif choice == '2':
        scene_path = input("Enter scene path: ").strip()
        nav_data_path = input("Enter navigation data JSON path: ").strip()
        sample_idx = int(input("Enter sample index (default 0): ").strip() or "0")
        
        if not Path(scene_path).exists():
            print(f"Error: Scene path not found: {scene_path}")
            sys.exit(1)
        
        if not Path(nav_data_path).exists():
            print(f"Error: Navigation data not found: {nav_data_path}")
            sys.exit(1)
        
        print(f"\nLoading sample {sample_idx}...")
        load_and_visualize_sample(scene_path, nav_data_path, sample_idx)
    
    else:
        print("Invalid choice")
