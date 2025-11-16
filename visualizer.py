#!/usr/bin/env python3
"""
Comprehensive Navigation Task Dataset Visualizer with Open3D GUI
Browse entire dataset: navigate across scenes, floors, and tasks
Features: Scene/floor selection, task navigation, point size slider, visibility toggles, label displays
"""

import json
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple


class DatasetVisualizer:
    """Interactive visualizer for browsing entire navigation task dataset"""
    
    # High-contrast colors for white background
    START_COLOR = [1.0, 0.0, 0.8]   # Bright Magenta
    END_COLOR = [0.0, 1.0, 0.0]     # Bright Green
    OBJECT_COLOR = [0.4, 0.4, 0.4]  # Dark gray
    REGION_COLOR = [0.6, 0.6, 0.6]  # Medium gray
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        
        # Scan dataset structure
        print("Scanning dataset...")
        self.scenes = self._scan_dataset()
        print(f"✓ Found {len(self.scenes)} scenes with {sum(len(floors) for floors in self.scenes.values())} total floors")
        
        # Current selection
        self.current_scene = list(self.scenes.keys())[0]
        self.current_floor_idx = 0
        self.current_task_idx = 0
        
        # Display settings
        self.show_point_cloud = True
        self.show_objects = True
        self.show_regions = True
        self.show_object_labels = False
        self.show_region_labels = False
        self.point_size = 2.0
        
        # Data containers
        self.objects = {}
        self.regions = {}
        self.tasks = []
        self.point_cloud = None
        
        # GUI components
        self.window = None
        self.scene_widget = None
        self.info_label = None
        self.scene_selector = None
        self.floor_selector = None
        self.task_label = None
        
        # 3D label tracking
        self.active_labels = []
        
        # Load initial data
        self._load_floor_data()
    
    def _scan_dataset(self) -> Dict[str, List[int]]:
        """Scan dataset directory to find all scenes and floors"""
        scenes = {}
        for scene_dir in sorted(self.dataset_dir.iterdir()):
            if scene_dir.is_dir() and not scene_dir.name.startswith('.'):
                floors = []
                for floor_dir in sorted(scene_dir.iterdir()):
                    if floor_dir.is_dir() and floor_dir.name.startswith('floor_'):
                        floor_num = int(floor_dir.name.split('_')[1])
                        floors.append(floor_num)
                if floors:
                    scenes[scene_dir.name] = sorted(floors)
        return scenes
    
    def _get_current_floor_dir(self) -> Path:
        """Get path to current floor directory"""
        floor_id = self.scenes[self.current_scene][self.current_floor_idx]
        return self.dataset_dir / self.current_scene / f'floor_{floor_id}'
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from current floor"""
        filepath = self._get_current_floor_dir() / filename
        if not filepath.exists():
            return {}
        with open(filepath) as f:
            return json.load(f)
    
    def _load_floor_data(self):
        """Load all data for current floor"""
        floor_dir = self._get_current_floor_dir()
        floor_id = self.scenes[self.current_scene][self.current_floor_idx]
        
        print(f"\nLoading: {self.current_scene} / Floor {floor_id}")
        
        # Load metadata
        self.objects = self._load_json(f'{self.current_scene}_floor{floor_id}_object_lookup.json')
        self.regions = self._load_json(f'{self.current_scene}_floor{floor_id}_region_lookup.json')
        
        # Load tasks
        task_file = floor_dir / 'navigation_tasks' / f'{self.current_scene}_floor{floor_id}_tasks.jsonl'
        self.tasks = []
        if task_file.exists():
            with open(task_file) as f:
                for line in f:
                    if line.strip():
                        self.tasks.append(json.loads(line))
        
        # Load point cloud
        floor_ply = floor_dir / f'{self.current_scene}_floor{floor_id}.ply'
        if floor_ply.exists():
            self.point_cloud = o3d.t.io.read_point_cloud(str(floor_ply))
        else:
            self.point_cloud = None
        
        # Reset task index
        self.current_task_idx = 0
        
        print(f"  ✓ {len(self.objects)} objects")
        print(f"  ✓ {len(self.regions)} regions")
        print(f"  ✓ {len(self.tasks)} tasks")
        if self.point_cloud:
            num_points = len(self.point_cloud.point.positions)
            print(f"  ✓ {num_points:,} points")
    
    def _create_bbox_lineset(self, center: np.ndarray, dimensions: np.ndarray, 
                             color: List[float], heading: float = 0.0) -> o3d.geometry.LineSet:
        """Create 3D bounding box as LineSet"""
        dx, dy, dz = dimensions / 2
        
        corners = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]
        ])
        
        if abs(heading) > 1e-6:
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
            corners = corners @ R.T
        
        corners += center
        
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(corners)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return lineset
    
    def _create_sphere_marker(self, position: np.ndarray, color: List[float], 
                              radius: float = 1.0) -> o3d.geometry.TriangleMesh:
        """Create large visible sphere marker"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere
    
    def create_window(self):
        """Create main GUI window"""
        gui.Application.instance.initialize()
        
        self.window = gui.Application.instance.create_window(
            f"Navigation Task Dataset Visualizer", 
            1920, 1080
        )
        
        self.em = self.window.theme.font_size
        
        # Create scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
        
        # Create control panel
        self.control_panel = gui.Vert(
            0.5 * self.em,
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em)
        )
        
        # === SCENE SELECTION ===
        self.control_panel.add_child(gui.Label("SCENE:"))
        self.scene_selector = gui.Combobox()
        for scene in self.scenes.keys():
            self.scene_selector.add_item(scene)
        self.scene_selector.selected_index = list(self.scenes.keys()).index(self.current_scene)
        self.scene_selector.set_on_selection_changed(self.on_scene_changed)
        self.control_panel.add_child(self.scene_selector)
        
        # === FLOOR SELECTION ===
        self.control_panel.add_child(gui.Label("FLOOR:"))
        self.floor_selector = gui.Combobox()
        self._update_floor_selector()
        self.floor_selector.set_on_selection_changed(self.on_floor_changed)
        self.control_panel.add_child(self.floor_selector)
        
        self.control_panel.add_fixed(0.5 * self.em)
        
        # === TASK INFO ===
        self.info_label = gui.Label("")
        self.control_panel.add_child(self.info_label)
        
        self.control_panel.add_fixed(0.5 * self.em)
        
        # === TASK NAVIGATION ===
        self.task_label = gui.Label("TASK:")
        self.control_panel.add_child(self.task_label)
        
        nav_horiz = gui.Horiz(0.5 * self.em)
        prev_button = gui.Button("◀ Prev")
        prev_button.set_on_clicked(self.on_prev_task)
        next_button = gui.Button("Next ▶")
        next_button.set_on_clicked(self.on_next_task)
        nav_horiz.add_child(prev_button)
        nav_horiz.add_stretch()
        nav_horiz.add_child(next_button)
        self.control_panel.add_child(nav_horiz)
        
        self.control_panel.add_fixed(0.5 * self.em)
        
        # === POINT SIZE SLIDER ===
        self.control_panel.add_child(gui.Label("Point Size:"))
        self.point_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.point_size_slider.set_limits(1.0, 20.0)
        self.point_size_slider.double_value = self.point_size
        self.point_size_slider.set_on_value_changed(self.on_point_size_changed)
        self.control_panel.add_child(self.point_size_slider)
        
        self.control_panel.add_fixed(0.5 * self.em)
        
        # === VISIBILITY TOGGLES ===
        self.control_panel.add_child(gui.Label("Visibility:"))
        
        self.pcd_checkbox = gui.Checkbox("Show Point Cloud")
        self.pcd_checkbox.checked = self.show_point_cloud
        self.pcd_checkbox.set_on_checked(self.on_pcd_visibility_changed)
        self.control_panel.add_child(self.pcd_checkbox)
        
        self.obj_checkbox = gui.Checkbox("Show Objects")
        self.obj_checkbox.checked = self.show_objects
        self.obj_checkbox.set_on_checked(self.on_obj_visibility_changed)
        self.control_panel.add_child(self.obj_checkbox)
        
        self.reg_checkbox = gui.Checkbox("Show Regions")
        self.reg_checkbox.checked = self.show_regions
        self.reg_checkbox.set_on_checked(self.on_reg_visibility_changed)
        self.control_panel.add_child(self.reg_checkbox)
        
        self.control_panel.add_fixed(0.5 * self.em)
        
        # === LABEL TOGGLES ===
        self.control_panel.add_child(gui.Label("Labels:"))
        
        self.obj_labels_checkbox = gui.Checkbox("Show Object Names")
        self.obj_labels_checkbox.checked = self.show_object_labels
        self.obj_labels_checkbox.set_on_checked(self.on_obj_labels_changed)
        self.control_panel.add_child(self.obj_labels_checkbox)
        
        self.reg_labels_checkbox = gui.Checkbox("Show Region Names")
        self.reg_labels_checkbox.checked = self.show_region_labels
        self.reg_labels_checkbox.set_on_checked(self.on_reg_labels_changed)
        self.control_panel.add_child(self.reg_labels_checkbox)
        
        # Add widgets to window
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.control_panel)
        
        # Set layout callback
        self.window.set_on_layout(self.on_layout)
        
        # Display initial scene
        self.update_scene()
    
    def _update_floor_selector(self):
        """Update floor selector dropdown with current scene's floors"""
        self.floor_selector.clear_items()
        for floor_id in self.scenes[self.current_scene]:
            self.floor_selector.add_item(f"Floor {floor_id}")
        self.floor_selector.selected_index = self.current_floor_idx
    
    def on_layout(self, layout_context):
        """Handle window layout"""
        content_rect = self.window.content_rect
        panel_width = 22 * self.em
        
        self.scene_widget.frame = gui.Rect(
            content_rect.x,
            content_rect.y,
            content_rect.width - panel_width,
            content_rect.height
        )
        
        self.control_panel.frame = gui.Rect(
            content_rect.width - panel_width,
            content_rect.y,
            panel_width,
            content_rect.height
        )
    
    def on_scene_changed(self, scene_name: str, idx: int):
        """Handle scene selection change"""
        self.current_scene = scene_name
        self.current_floor_idx = 0
        self._update_floor_selector()
        self._load_floor_data()
        self.update_scene()
    
    def on_floor_changed(self, floor_name: str, idx: int):
        """Handle floor selection change"""
        self.current_floor_idx = idx
        self._load_floor_data()
        self.update_scene()
    
    def on_prev_task(self):
        """Go to previous task"""
        if self.current_task_idx > 0:
            self.current_task_idx -= 1
            self.update_scene()
    
    def on_next_task(self):
        """Go to next task"""
        if self.current_task_idx < len(self.tasks) - 1:
            self.current_task_idx += 1
            self.update_scene()
    
    def on_point_size_changed(self, new_value):
        """Handle point size slider change"""
        self.point_size = new_value
        self.update_scene()
    
    def on_pcd_visibility_changed(self, is_checked):
        """Handle point cloud visibility toggle"""
        self.show_point_cloud = is_checked
        self.update_scene()
    
    def on_obj_visibility_changed(self, is_checked):
        """Handle object visibility toggle"""
        self.show_objects = is_checked
        self.update_scene()
    
    def on_reg_visibility_changed(self, is_checked):
        """Handle region visibility toggle"""
        self.show_regions = is_checked
        self.update_scene()
    
    def on_obj_labels_changed(self, is_checked):
        """Handle object labels toggle"""
        self.show_object_labels = is_checked
        self.update_scene()
    
    def on_reg_labels_changed(self, is_checked):
        """Handle region labels toggle"""
        self.show_region_labels = is_checked
        self.update_scene()
    
    def update_scene(self):
        """Update 3D scene with current task"""
        if not self.tasks:
            self.info_label.text = "NO TASKS AVAILABLE"
            self.task_label.text = "TASK: 0/0"
            return
        
        task = self.tasks[self.current_task_idx]
        
        # Update task label
        self.task_label.text = f"TASK: {self.current_task_idx + 1}/{len(self.tasks)}"
        
        # Update info label
        info_text = f"INSTRUCTION:\n{task['instruction']}\n\n"
        info_text += f"START (Magenta):\n{task['start_point']['description']}\n\n"
        info_text += f"END (Green):\n{task['end_point']['description']}\n\n"
        info_text += f"Distance: {task['spatial_metadata']['euclidean_distance']:.2f}m\n"
        info_text += f"Difficulty: {task['spatial_metadata']['difficulty'].upper()}"
        self.info_label.text = info_text
        
        # Clear scene and labels
        self.scene_widget.scene.clear_geometry()
        for label in self.active_labels:
            self.scene_widget.remove_3d_label(label)
        self.active_labels.clear()
        
        # Extract IDs (convert to string for dict lookup)
        start_id = str(task['start_point']['id'])
        start_type = task['start_point']['type']
        end_id = str(task['end_point']['id'])
        end_type = task['end_point']['type']
        
        # Add point cloud
        if self.show_point_cloud and self.point_cloud is not None:
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = self.point_size
            self.scene_widget.scene.add_geometry("point_cloud", self.point_cloud, mat)
        
        # Add regions
        if self.show_regions:
            for region_id, region in self.regions.items():
                if region.get('position') and region.get('bbox_dimensions'):
                    bbox = self._create_bbox_lineset(
                        np.array(region['position']),
                        np.array(region['bbox_dimensions']),
                        self.REGION_COLOR,
                        0.0
                    )
                    mat = rendering.MaterialRecord()
                    mat.shader = "unlitLine"
                    mat.line_width = 2.0
                    self.scene_widget.scene.add_geometry(f"region_{region_id}", bbox, mat)
                    
                    # Add region label
                    if self.show_region_labels:
                        label = self.scene_widget.add_3d_label(
                            np.array(region['position']),
                            f"{region.get('region_type', 'region')} ({region_id})"
                        )
                        label.color = gui.Color(0.6, 0.6, 0.6)
                        label.scale = 1.2
                        self.active_labels.append(label)
        
        # Add objects
        if self.show_objects:
            for obj_id, obj in self.objects.items():
                if obj_id == start_id or obj_id == end_id:
                    continue
                
                if obj.get('position') and obj.get('bbox_dimensions'):
                    bbox = self._create_bbox_lineset(
                        np.array(obj['position']),
                        np.array(obj['bbox_dimensions']),
                        self.OBJECT_COLOR,
                        obj.get('heading', 0.0)
                    )
                    mat = rendering.MaterialRecord()
                    mat.shader = "unlitLine"
                    mat.line_width = 2.0
                    self.scene_widget.scene.add_geometry(f"object_{obj_id}", bbox, mat)
                    
                    # Add object label
                    if self.show_object_labels:
                        label = self.scene_widget.add_3d_label(
                            np.array(obj['position']),
                            f"{obj.get('label', 'object')} ({obj_id})"
                        )
                        label.color = gui.Color(0.4, 0.4, 0.4)
                        label.scale = 1.0
                        self.active_labels.append(label)
        
        # Add START marker (MAGENTA)
        if start_type == 'object' and start_id in self.objects:
            obj = self.objects[start_id]
            if obj.get('position'):
                # Large sphere
                sphere = self._create_sphere_marker(np.array(obj['position']), self.START_COLOR, radius=1.0)
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                self.scene_widget.scene.add_geometry("start_sphere", sphere, mat)
                
                # Bounding box
                if obj.get('bbox_dimensions'):
                    bbox = self._create_bbox_lineset(
                        np.array(obj['position']),
                        np.array(obj['bbox_dimensions']),
                        self.START_COLOR,
                        obj.get('heading', 0.0)
                    )
                    mat = rendering.MaterialRecord()
                    mat.shader = "unlitLine"
                    mat.line_width = 5.0
                    self.scene_widget.scene.add_geometry("start_bbox", bbox, mat)
                
                # Label
                label = self.scene_widget.add_3d_label(
                    np.array(obj['position']),
                    f"START: {obj.get('label', 'object')} ({start_id})"
                )
                label.color = gui.Color(1.0, 0.0, 0.8)
                label.scale = 1.5
                self.active_labels.append(label)
        
        elif start_type == 'region' and start_id in self.regions:
            region = self.regions[start_id]
            if region.get('position'):
                sphere = self._create_sphere_marker(np.array(region['position']), self.START_COLOR, radius=1.0)
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                self.scene_widget.scene.add_geometry("start_sphere", sphere, mat)
                
                if region.get('bbox_dimensions'):
                    bbox = self._create_bbox_lineset(
                        np.array(region['position']),
                        np.array(region['bbox_dimensions']),
                        self.START_COLOR,
                        0.0
                    )
                    mat = rendering.MaterialRecord()
                    mat.shader = "unlitLine"
                    mat.line_width = 5.0
                    self.scene_widget.scene.add_geometry("start_bbox", bbox, mat)
                
                label = self.scene_widget.add_3d_label(
                    np.array(region['position']),
                    f"START: {region.get('region_type', 'region')} ({start_id})"
                )
                label.color = gui.Color(1.0, 0.0, 0.8)
                label.scale = 1.5
                self.active_labels.append(label)
        
        # Add END marker (GREEN)
        if end_type == 'object' and end_id in self.objects:
            obj = self.objects[end_id]
            if obj.get('position'):
                sphere = self._create_sphere_marker(np.array(obj['position']), self.END_COLOR, radius=1.0)
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                self.scene_widget.scene.add_geometry("end_sphere", sphere, mat)
                
                if obj.get('bbox_dimensions'):
                    bbox = self._create_bbox_lineset(
                        np.array(obj['position']),
                        np.array(obj['bbox_dimensions']),
                        self.END_COLOR,
                        obj.get('heading', 0.0)
                    )
                    mat = rendering.MaterialRecord()
                    mat.shader = "unlitLine"
                    mat.line_width = 5.0
                    self.scene_widget.scene.add_geometry("end_bbox", bbox, mat)
                
                label = self.scene_widget.add_3d_label(
                    np.array(obj['position']),
                    f"END: {obj.get('label', 'object')} ({end_id})"
                )
                label.color = gui.Color(0.0, 1.0, 0.0)
                label.scale = 1.5
                self.active_labels.append(label)
        
        elif end_type == 'region' and end_id in self.regions:
            region = self.regions[end_id]
            if region.get('position'):
                sphere = self._create_sphere_marker(np.array(region['position']), self.END_COLOR, radius=1.0)
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                self.scene_widget.scene.add_geometry("end_sphere", sphere, mat)
                
                if region.get('bbox_dimensions'):
                    bbox = self._create_bbox_lineset(
                        np.array(region['position']),
                        np.array(region['bbox_dimensions']),
                        self.END_COLOR,
                        0.0
                    )
                    mat = rendering.MaterialRecord()
                    mat.shader = "unlitLine"
                    mat.line_width = 5.0
                    self.scene_widget.scene.add_geometry("end_bbox", bbox, mat)
                
                label = self.scene_widget.add_3d_label(
                    np.array(region['position']),
                    f"END: {region.get('region_type', 'region')} ({end_id})"
                )
                label.color = gui.Color(0.0, 1.0, 0.0)
                label.scale = 1.5
                self.active_labels.append(label)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene_widget.scene.add_geometry("coord_frame", coord_frame, mat)
        
        # Set white background
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        
        # Setup camera
        if self.point_cloud is not None:
            bounds = self.point_cloud.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60, bounds.to_legacy(), bounds.get_center().numpy())
        
        self.window.post_redraw()
    
    def run(self):
        """Run the GUI application"""
        gui.Application.instance.run()


def main():
    parser = argparse.ArgumentParser(description="Navigation Task Dataset Visualizer")
    parser.add_argument('--dataset_dir', type=str, required=True, 
                       help='Path to dataset directory (e.g., output/floors)')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Navigation Task Dataset Visualizer")
    print(f"{'='*80}\n")
    
    visualizer = DatasetVisualizer(Path(args.dataset_dir))
    visualizer.create_window()
    visualizer.run()


if __name__ == '__main__':
    main()
