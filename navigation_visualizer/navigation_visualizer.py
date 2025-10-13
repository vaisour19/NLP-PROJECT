#!/usr/bin/env python3
"""
PyVista-based 3D Navigation Dataset Visualizer

Visualizes synthetic navigation data with:
- Scene point clouds or meshes
- Region bounding boxes (translucent)
- Object bounding boxes (translucent)
- Navigation trajectories
- Action arrows (north/south/east/west/up/down)
- Start/goal markers
- Interactive controls with transparency sliders
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton,
                             QGroupBox, QScrollArea, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt


class NavigationVisualizer:
    """PyVista-based visualizer for synthetic navigation data"""
    
    def __init__(self, 
                 scene_path: str,
                 navigation_data_path: str,
                 step_size: float = 1.0):
        """
        Args:
            scene_path: Path to scene directory (contains CSVs, point cloud)
            navigation_data_path: Path to navigation JSON file
            step_size: Distance per action (for arrow scaling)
        """
        self.scene_path = Path(scene_path)
        self.scene_name = self.scene_path.name
        self.step_size = step_size
        
        # File paths
        self.pcd_path = self.scene_path / f'{self.scene_name}_pc_result.ply'
        self.region_csv = self.scene_path / f'{self.scene_name}_region_result.csv'
        self.object_csv = self.scene_path / f'{self.scene_name}_object_result.csv'
        
        # Load data
        print(f"Loading scene: {self.scene_name}")
        self.load_scene_data()
        self.load_navigation_data(navigation_data_path)
        
        # Visualization state
        self.current_sample_idx = 0
        self.actors = {}  # Store actors for updating
        
        # Transparency settings
        self.transparency = {
            'regions': 0.1,
            'objects': 0.15,  # Lower opacity for objects
            'trajectory': 1.0,
            'arrows': 0.8,
            'point_cloud': 1.0
        }
        
        # Visibility settings
        self.visibility = {
            'regions': True,
            'objects': True,
            'trajectory': True,
            'arrows': True,
            'point_cloud': True,
            'start_marker': True,
            'goal_marker': True,
            'waypoints': True
        }
        
        # Colors
        self.colors = {
            'start': '#00FF00',      # Green
            'goal': '#FF0000',       # Red
            'trajectory': '#FFFF00', # Yellow
            'waypoint': '#00FFFF',   # Cyan
            'region': '#4A90E2',     # Blue
            'object': '#F5A623',     # Orange
            'north': '#FF0000',      # Red
            'south': '#0000FF',      # Blue
            'east': '#00FF00',       # Green
            'west': '#FFFF00',       # Yellow
            'up': '#FF00FF',         # Magenta
            'down': '#00FFFF'        # Cyan
        }
        
        # Initialize plotter
        self.init_plotter()
    
    def load_scene_data(self):
        """Load scene regions and objects"""
        # Load regions
        self.region_df = pd.read_csv(self.region_csv)
        print(f"  Loaded {len(self.region_df)} regions")
        
        # Load objects
        self.object_df = pd.read_csv(self.object_csv)
        # Filter out non-navigable objects
        non_navigable = ['wall', 'ceiling', 'floor', 'void', 'remove']
        self.object_df = self.object_df[~self.object_df['nyu40_label'].isin(non_navigable)]
        print(f"  Loaded {len(self.object_df)} navigable objects")
        
        # Load point cloud if available
        self.point_cloud = None
        if self.pcd_path.exists():
            try:
                self.point_cloud = pv.read(str(self.pcd_path))
                print(f"  Loaded point cloud: {self.point_cloud.n_points} points")
            except Exception as e:
                print(f"  Warning: Could not load point cloud: {e}")
    
    def load_navigation_data(self, nav_path: str):
        """Load navigation dataset JSON"""
        with open(nav_path, 'r') as f:
            data = json.load(f)
        
        self.nav_data = data
        self.samples = data['samples']
        print(f"  Loaded {len(self.samples)} navigation samples")
        
        if self.samples:
            sample = self.samples[0]
            print(f"\nSample structure:")
            print(f"  Task ID: {sample.get('task_id')}")
            print(f"  Instruction: {sample.get('instruction')[:80]}...")
            print(f"  Actions: {len(sample.get('action_sequence', []))} actions")
            print(f"  Path length: {sample.get('path_length', 0):.2f}m")
    
    def init_plotter(self):
        """Initialize PyVista plotter with Qt backend"""
        self.plotter = BackgroundPlotter(
            title=f"Navigation Visualizer - {self.scene_name}",
            window_size=(1600, 1000)
        )
        
        # Set background
        self.plotter.set_background('white')
        
        # Add axes
        self.plotter.add_axes(
            xlabel='X (East)',
            ylabel='Y (North)',
            zlabel='Z (Up)'
        )
        
        # Create control panel
        self.create_control_panel()
        
        # Initial visualization (moved after control panel is fully created)
        # Will be called at the end of create_control_panel()
    
    def create_control_panel(self):
        """Create Qt control panel with sliders and buttons"""
        # Access the main window
        main_window = self.plotter.app_window
        
        # Create control widget
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        
        # Sample selector
        sample_group = QGroupBox("Navigation Sample")
        sample_layout = QVBoxLayout()
        
        sample_info_layout = QHBoxLayout()
        sample_info_layout.addWidget(QLabel("Sample:"))
        self.sample_spinbox = QSpinBox()
        self.sample_spinbox.setMinimum(0)
        self.sample_spinbox.setMaximum(len(self.samples) - 1)
        self.sample_spinbox.setValue(0)
        self.sample_spinbox.valueChanged.connect(self.on_sample_changed)
        sample_info_layout.addWidget(self.sample_spinbox)
        sample_info_layout.addWidget(QLabel(f"/ {len(self.samples) - 1}"))
        sample_layout.addLayout(sample_info_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◀ Previous")
        prev_btn.clicked.connect(self.previous_sample)
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self.next_sample)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        sample_layout.addLayout(nav_layout)
        
        # Instruction display
        self.instruction_label = QLabel()
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        sample_layout.addWidget(QLabel("Instruction:"))
        sample_layout.addWidget(self.instruction_label)
        
        # Sample stats
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        sample_layout.addWidget(QLabel("Statistics:"))
        sample_layout.addWidget(self.stats_label)
        
        sample_group.setLayout(sample_layout)
        control_layout.addWidget(sample_group)
        
        # Visibility controls
        visibility_group = QGroupBox("Visibility")
        visibility_layout = QVBoxLayout()
        
        self.visibility_checkboxes = {}
        for key, label in [
            ('point_cloud', 'Point Cloud'),
            ('regions', 'Region Boxes'),
            ('objects', 'Object Boxes'),
            ('trajectory', 'Trajectory'),
            ('waypoints', 'Waypoints'),
            ('arrows', 'Action Arrows'),
            ('start_marker', 'Start Marker'),
            ('goal_marker', 'Goal Marker')
        ]:
            cb = QCheckBox(label)
            cb.setChecked(self.visibility[key])
            cb.stateChanged.connect(lambda state, k=key: self.on_visibility_changed(k, state))
            visibility_layout.addWidget(cb)
            self.visibility_checkboxes[key] = cb
        
        visibility_group.setLayout(visibility_layout)
        control_layout.addWidget(visibility_group)
        
        # Transparency sliders
        transparency_group = QGroupBox("Transparency")
        transparency_layout = QVBoxLayout()
        
        self.transparency_sliders = {}
        for key, label in [
            ('point_cloud', 'Point Cloud'),
            ('regions', 'Region Boxes'),
            ('objects', 'Object Boxes'),
            ('trajectory', 'Trajectory'),
            ('arrows', 'Action Arrows')
        ]:
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(label))
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(self.transparency[key] * 100))
            slider.valueChanged.connect(lambda val, k=key: self.on_transparency_changed(k, val))
            slider_layout.addWidget(slider)
            transparency_layout.addLayout(slider_layout)
            self.transparency_sliders[key] = slider
        
        transparency_group.setLayout(transparency_layout)
        control_layout.addWidget(transparency_group)
        
        # Add stretch to push everything to top
        control_layout.addStretch()
        
        control_widget.setLayout(control_layout)
        
        # Add control widget to main window in a dock
        from PyQt5.QtWidgets import QDockWidget
        from PyQt5.QtCore import Qt as QtCore
        
        dock = QDockWidget("Controls", main_window)
        dock.setWidget(control_widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        
        # Add dock to right side of main window
        main_window.addDockWidget(QtCore.RightDockWidgetArea, dock)
        
        main_window.signal_close.connect(self.plotter.close)
        main_window.setWindowTitle(f"Navigation Visualizer - {self.scene_name}")
        
        # Now that all widgets are created, do initial visualization
        self.update_visualization()
    
    def on_sample_changed(self, value):
        """Handle sample selection change"""
        self.current_sample_idx = value
        self.update_visualization()
    
    def previous_sample(self):
        """Go to previous sample"""
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.sample_spinbox.setValue(self.current_sample_idx)
    
    def next_sample(self):
        """Go to next sample"""
        if self.current_sample_idx < len(self.samples) - 1:
            self.current_sample_idx += 1
            self.sample_spinbox.setValue(self.current_sample_idx)
    
    def on_visibility_changed(self, key: str, state):
        """Handle visibility checkbox change"""
        self.visibility[key] = (state == Qt.Checked)
        self.update_visualization()
    
    def on_transparency_changed(self, key: str, value):
        """Handle transparency slider change"""
        self.transparency[key] = value / 100.0
        self.update_actor_transparency(key)
    
    def update_actor_transparency(self, key: str):
        """Update transparency for specific actor type"""
        opacity = self.transparency[key]
        
        if key in self.actors:
            actors = self.actors[key]
            if isinstance(actors, list):
                for actor in actors:
                    if actor is not None:
                        actor.GetProperty().SetOpacity(opacity)
            elif actors is not None:
                actors.GetProperty().SetOpacity(opacity)
        
        self.plotter.render()
    
    def update_visualization(self):
        """Update entire visualization for current sample"""
        # Clear previous actors
        self.plotter.clear()
        self.actors = {}
        
        # Get current sample
        if not self.samples:
            return
        
        sample = self.samples[self.current_sample_idx]
        
        # Update UI
        self.instruction_label.setText(sample['instruction'])
        stats_text = (
            f"Task: {sample['task_id']}\n"
            f"Actions: {sample['num_actions']}\n"
            f"Path Length: {sample['path_length']:.2f}m\n"
            f"Vertical: {'Yes' if sample['has_vertical_movement'] else 'No'}\n"
            f"Start Region: {sample['start_region']}\n"
            f"Goal Region: {sample['goal_region']}"
        )
        self.stats_label.setText(stats_text)
        
        # Add point cloud with RGB colors
        if self.visibility['point_cloud'] and self.point_cloud is not None:
            # Check if point cloud has RGB data
            has_rgb = 'RGB' in self.point_cloud.array_names or 'rgb' in self.point_cloud.array_names
            
            if has_rgb:
                # Use RGB colors from PLY file
                rgb_name = 'RGB' if 'RGB' in self.point_cloud.array_names else 'rgb'
                actor = self.plotter.add_mesh(
                    self.point_cloud,
                    scalars=rgb_name,
                    rgb=True,  # Interpret as RGB values
                    opacity=self.transparency['point_cloud'],
                    point_size=2,
                    render_points_as_spheres=False  # Faster rendering
                )
            else:
                # Fallback to single color if no RGB data
                actor = self.plotter.add_mesh(
                    self.point_cloud,
                    color='lightgray',
                    opacity=self.transparency['point_cloud'],
                    point_size=2,
                    render_points_as_spheres=False
                )
            self.actors['point_cloud'] = actor
        
        # Add regions
        if self.visibility['regions']:
            self.add_region_boxes()
        
        # Add objects
        if self.visibility['objects']:
            self.add_object_boxes()
        
        # Add trajectory
        if self.visibility['trajectory']:
            self.add_trajectory(sample)
        
        # Add waypoints
        if self.visibility['waypoints']:
            self.add_waypoints(sample)
        
        # Add action arrows
        if self.visibility['arrows']:
            self.add_action_arrows(sample)
        
        # Add start marker
        if self.visibility['start_marker']:
            self.add_start_marker(sample)
        
        # Add goal marker
        if self.visibility['goal_marker']:
            self.add_goal_marker(sample)
        
        # Reset camera to fit scene
        self.plotter.reset_camera()
    
    def add_region_boxes(self):
        """Add region bounding boxes"""
        actors = []
        for _, row in self.region_df.iterrows():
            center = np.array([
                row['region_bbox_cx'],
                row['region_bbox_cy'],
                row['region_bbox_cz']
            ])
            size = np.array([
                row['region_bbox_xlength'],
                row['region_bbox_ylength'],
                row['region_bbox_zlength']
            ])
            
            box = pv.Cube(
                center=center,
                x_length=size[0],
                y_length=size[1],
                z_length=size[2]
            )
            
            actor = self.plotter.add_mesh(
                box,
                color=self.colors['region'],
                opacity=self.transparency['regions'],
                style='wireframe',
                line_width=2,
                label=f"Region: {row['region_label']}"
            )
            actors.append(actor)
        
        self.actors['regions'] = actors
    
    def add_object_boxes(self):
        """Add object bounding boxes"""
        actors = []
        for _, row in self.object_df.iterrows():
            center = np.array([
                row['object_bbox_cx'],
                row['object_bbox_cy'],
                row['object_bbox_cz']
            ])
            size = np.array([
                row['object_bbox_xlength'],
                row['object_bbox_ylength'],
                row['object_bbox_zlength']
            ])
            
            box = pv.Cube(
                center=center,
                x_length=size[0],
                y_length=size[1],
                z_length=size[2]
            )
            
            actor = self.plotter.add_mesh(
                box,
                color=self.colors['object'],
                opacity=self.transparency['objects'],
                style='wireframe',  # Changed from 'surface' to 'wireframe'
                line_width=1.5,
                label=f"Object: {row['nyu40_label']}"
            )
            actors.append(actor)
        
        self.actors['objects'] = actors
    
    def reconstruct_path_from_actions(self, sample: Dict) -> List[np.ndarray]:
        """Reconstruct full step-by-step path from action sequence
        
        This creates waypoints for EVERY action, showing the actual navigation
        path through the scene (including stairs, corridors, etc.)
        """
        coords = sample['path_coordinates']
        actions = sample['action_sequence']
        
        if not coords or len(coords) < 1:
            return []
        
        # Start at the first coordinate
        full_path = [np.array(coords[0])]
        current_pos = np.array(coords[0]).copy()
        
        # Action direction vectors
        action_vectors = {
            'north': np.array([0, self.step_size, 0]),
            'south': np.array([0, -self.step_size, 0]),
            'east': np.array([self.step_size, 0, 0]),
            'west': np.array([-self.step_size, 0, 0]),
            'up': np.array([0, 0, self.step_size]),
            'down': np.array([0, 0, -self.step_size]),
            'stop': np.array([0, 0, 0])
        }
        
        # Execute each action to build full path
        for action in actions:
            if action == 'stop':
                break
            
            if action in action_vectors:
                current_pos = current_pos + action_vectors[action]
                full_path.append(current_pos.copy())
        
        return full_path
    
    def add_trajectory(self, sample: Dict):
        """Add trajectory path as a tube"""
        # Reconstruct full path from actions (shows actual navigation)
        full_path = self.reconstruct_path_from_actions(sample)
        
        if len(full_path) < 2:
            return
        
        coords = np.array(full_path)
        
        # Create line from coordinates
        points = pv.PolyData(coords)
        
        # Create spline for smooth path
        spline = pv.Spline(coords, max(len(coords) * 2, 50))
        
        # Add as tube
        tube = spline.tube(radius=0.05)
        actor = self.plotter.add_mesh(
            tube,
            color=self.colors['trajectory'],
            opacity=self.transparency['trajectory'],
            label='Trajectory'
        )
        
        self.actors['trajectory'] = actor
    
    def add_waypoints(self, sample: Dict):
        """Add waypoint spheres at key positions along the path"""
        # Use reconstructed full path
        full_path = self.reconstruct_path_from_actions(sample)
        
        if len(full_path) < 2:
            return
        
        actors = []
        # Show waypoints every N steps to avoid clutter
        waypoint_interval = max(3, len(full_path) // 10)
        
        for i in range(0, len(full_path), waypoint_interval):
            if i == 0 or i == len(full_path) - 1:
                continue  # Skip start/goal (handled separately)
            
            sphere = pv.Sphere(radius=0.1, center=full_path[i])
            actor = self.plotter.add_mesh(
                sphere,
                color=self.colors['waypoint'],
                opacity=0.8,
                label=f'Waypoint {i}'
            )
            actors.append(actor)
        
        self.actors['waypoints'] = actors
    
    def add_action_arrows(self, sample: Dict):
        """Add action arrows showing navigation directions along the path"""
        actions = sample['action_sequence']
        
        # Get full reconstructed path
        full_path = self.reconstruct_path_from_actions(sample)
        
        if len(full_path) < 2:
            return
        
        actors = []
        
        # Action directions (normalized for arrow visualization)
        action_vectors = {
            'north': np.array([0, 1, 0]),
            'south': np.array([0, -1, 0]),
            'east': np.array([1, 0, 0]),
            'west': np.array([-1, 0, 0]),
            'up': np.array([0, 0, 1]),
            'down': np.array([0, 0, -1]),
            'stop': np.array([0, 0, 0])
        }
        
        # Place arrows along the reconstructed path
        # Show every Nth action to avoid clutter
        arrow_interval = max(2, len(actions) // 15)  # Show ~15 arrows max
        
        for i, action in enumerate(actions):
            if action == 'stop':
                break
            
            if action not in action_vectors:
                continue
            
            # Only show arrows at intervals
            if i % arrow_interval != 0:
                continue
            
            # Position is at the path index
            if i >= len(full_path):
                break
                
            current_pos = full_path[i]
            direction = action_vectors[action]
            
            if np.linalg.norm(direction) == 0:
                continue
            
            # Create arrow
            arrow = pv.Arrow(
                start=current_pos,
                direction=direction,
                scale=self.step_size * 0.7,
                tip_length=0.25,
                tip_radius=0.12,
                shaft_radius=0.04
            )
            
            actor = self.plotter.add_mesh(
                arrow,
                color=self.colors.get(action, '#888888'),
                opacity=self.transparency['arrows'],
                label=f'Action: {action}'
            )
            actors.append(actor)
        
        self.actors['arrows'] = actors
    
    def add_start_marker(self, sample: Dict):
        """Add start position marker"""
        coords = np.array(sample['path_coordinates'])
        if len(coords) == 0:
            return
        
        start_pos = coords[0]
        
        # Large sphere
        sphere = pv.Sphere(radius=0.25, center=start_pos)
        actor = self.plotter.add_mesh(
            sphere,
            color=self.colors['start'],
            opacity=0.9,
            label='Start'
        )
        
        # Add text label
        self.plotter.add_point_labels(
            [start_pos],
            ['START'],
            font_size=20,
            point_color=self.colors['start'],
            text_color='black',
            shape_opacity=0.7
        )
        
        self.actors['start_marker'] = actor
    
    def add_goal_marker(self, sample: Dict):
        """Add goal position marker"""
        coords = np.array(sample['path_coordinates'])
        if len(coords) == 0:
            return
        
        goal_pos = coords[-1]
        
        # Large sphere
        sphere = pv.Sphere(radius=0.25, center=goal_pos)
        actor = self.plotter.add_mesh(
            sphere,
            color=self.colors['goal'],
            opacity=0.9,
            label='Goal'
        )
        
        # Add text label
        self.plotter.add_point_labels(
            [goal_pos],
            ['GOAL'],
            font_size=20,
            point_color=self.colors['goal'],
            text_color='black',
            shape_opacity=0.7
        )
        
        self.actors['goal_marker'] = actor
    
    def show(self):
        """Show the visualizer and keep it running"""
        # BackgroundPlotter handles its own Qt event loop
        # The app_window will stay open until closed by user
        if hasattr(self.plotter, 'app_window') and self.plotter.app_window:
            # Keep reference to prevent garbage collection
            self.app_window = self.plotter.app_window
            # The BackgroundPlotter's Qt application runs in its own event loop
            # and keeps the window open automatically


def main():
    parser = argparse.ArgumentParser(
        description='Visualize synthetic navigation data with PyVista'
    )
    
    parser.add_argument(
        '--scene_path',
        type=str,
        required=True,
        help='Path to scene directory (contains CSVs and point cloud)'
    )
    
    parser.add_argument(
        '--navigation_data',
        type=str,
        required=True,
        help='Path to navigation JSON file'
    )
    
    parser.add_argument(
        '--step_size',
        type=float,
        default=1.0,
        help='Distance per action (default: 1.0m)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    scene_path = Path(args.scene_path)
    if not scene_path.exists():
        print(f"Error: Scene path does not exist: {scene_path}")
        return 1
    
    nav_path = Path(args.navigation_data)
    if not nav_path.exists():
        print(f"Error: Navigation data file does not exist: {nav_path}")
        return 1
    
    print(f"\n{'='*80}")
    print("Navigation Dataset Visualizer (PyVista)")
    print(f"{'='*80}\n")
    
    # Create visualizer
    visualizer = NavigationVisualizer(
        scene_path=str(scene_path),
        navigation_data_path=str(nav_path),
        step_size=args.step_size
    )
    
    print("\n✓ Visualizer launched successfully!")
    print("  - Use the control panel on the right to adjust visualization")
    print("  - Navigate between samples using Previous/Next buttons")
    print("  - Adjust transparency with sliders")
    print("  - Close the window to exit\n")
    
    # Start the Qt event loop to keep the window open
    # BackgroundPlotter uses its own Qt application
    visualizer.plotter.app.exec_()


if __name__ == '__main__':
    sys.exit(main())
