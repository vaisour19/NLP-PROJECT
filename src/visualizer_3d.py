"""
Interactive 3D visualization using Open3D
"""

import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from .floor_detector import FloorSlab
except ImportError:
    from floor_detector import FloorSlab


class FloorVisualizer3D:
    """Interactive 3D visualization of floor decomposition"""
    
    # Color palette for floors (RGB)
    FLOOR_COLORS = [
        [0.2, 0.6, 0.9],  # Blue
        [0.9, 0.4, 0.2],  # Orange
        [0.3, 0.8, 0.3],  # Green
        [0.9, 0.7, 0.2],  # Yellow
        [0.7, 0.2, 0.8],  # Purple
        [0.2, 0.8, 0.8],  # Cyan
        [0.9, 0.3, 0.5],  # Pink
        [0.5, 0.5, 0.5],  # Gray
    ]
    
    @staticmethod
    def create_bbox_lineset(center: np.ndarray, extents: np.ndarray, 
                           color: List[float], heading: float = 0.0) -> o3d.geometry.LineSet:
        """
        Create a 3D bounding box as a LineSet
        
        Args:
            center: [x, y, z] center of bbox
            extents: [dx, dy, dz] dimensions
            color: RGB color [0-1]
            heading: Rotation around Z-axis in radians
            
        Returns:
            Open3D LineSet representing the bbox
        """
        # Half extents
        dx, dy, dz = extents / 2
        
        # 8 corners of bbox (before rotation)
        corners = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],  # bottom
            [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]   # top
        ])
        
        # Rotation matrix around Z-axis
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rot_matrix = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ])
        
        # Rotate and translate
        corners = corners @ rot_matrix.T + center
        
        # Define 12 edges (lines) of the bbox
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Create LineSet
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(corners)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return lineset
    
    @staticmethod
    def create_floor_slab_mesh(floor: FloorSlab, color: List[float], 
                               xy_bounds: Tuple[float, float, float, float],
                               alpha: float = 0.3) -> o3d.geometry.TriangleMesh:
        """
        Create a semi-transparent floor slab mesh
        
        Args:
            floor: FloorSlab object
            color: RGB color [0-1]
            xy_bounds: (x_min, x_max, y_min, y_max) spatial bounds
            alpha: Transparency (not directly supported, visual only)
            
        Returns:
            Open3D TriangleMesh for the floor slab
        """
        x_min, x_max, y_min, y_max = xy_bounds
        z_center = floor.z_center
        thickness = 0.05  # Thin slab
        
        # 8 vertices of the slab
        vertices = np.array([
            [x_min, y_min, z_center - thickness],
            [x_max, y_min, z_center - thickness],
            [x_max, y_max, z_center - thickness],
            [x_min, y_max, z_center - thickness],
            [x_min, y_min, z_center + thickness],
            [x_max, y_min, z_center + thickness],
            [x_max, y_max, z_center + thickness],
            [x_min, y_max, z_center + thickness],
        ])
        
        # 12 triangles (2 per face, 6 faces)
        triangles = np.array([
            # Bottom face
            [0, 1, 2], [0, 2, 3],
            # Top face
            [4, 6, 5], [4, 7, 6],
            # Front face
            [0, 5, 1], [0, 4, 5],
            # Back face
            [2, 7, 3], [2, 6, 7],
            # Left face
            [0, 7, 4], [0, 3, 7],
            # Right face
            [1, 6, 2], [1, 5, 6]
        ])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        
        return mesh
    
    @staticmethod
    def visualize_scene_floors(scene_name: str, region_csv: str, object_csv: str,
                               floors: List[FloorSlab], interactive: bool = True,
                               save_path: str = None):
        """
        Create interactive 3D visualization of floors
        
        Args:
            scene_name: Scene identifier
            region_csv: Path to region CSV
            object_csv: Path to object CSV
            floors: List of detected floors
            interactive: If True, opens interactive viewer; if False, saves only
            save_path: Optional path to save screenshot
        """
        # Load data
        df_regions = pd.read_csv(region_csv)
        df_objects = pd.read_csv(object_csv)
        
        # Compute spatial bounds
        x_coords = df_regions['region_bbox_cx'].values
        y_coords = df_regions['region_bbox_cy'].values
        x_lengths = df_regions['region_bbox_xlength'].values
        y_lengths = df_regions['region_bbox_ylength'].values
        
        x_min = (x_coords - x_lengths / 2).min()
        x_max = (x_coords + x_lengths / 2).max()
        y_min = (y_coords - y_lengths / 2).min()
        y_max = (y_coords + y_lengths / 2).max()
        
        xy_bounds = (x_min, x_max, y_min, y_max)
        
        # Create geometry list
        geometries = []
        
        # Create floor assignment mapping
        region_to_floor = {}
        for floor in floors:
            for region_id in floor.region_ids:
                region_to_floor[region_id] = floor.floor_id
        
        # Add region bboxes colored by floor
        for idx, row in df_regions.iterrows():
            region_id = int(row['region_id'])
            floor_id = region_to_floor.get(region_id, -1)
            
            if floor_id == -1:
                continue  # Skip unassigned regions
            
            color = FloorVisualizer3D.FLOOR_COLORS[floor_id % len(FloorVisualizer3D.FLOOR_COLORS)]
            
            center = np.array([
                row['region_bbox_cx'],
                row['region_bbox_cy'],
                row['region_bbox_cz']
            ])
            
            extents = np.array([
                row['region_bbox_xlength'],
                row['region_bbox_ylength'],
                row['region_bbox_zlength']
            ])
            
            heading = row['region_bbox_heading']
            
            bbox_lineset = FloorVisualizer3D.create_bbox_lineset(
                center, extents, color, heading
            )
            geometries.append(bbox_lineset)
        
        # Add floor slabs
        for floor in floors:
            color = FloorVisualizer3D.FLOOR_COLORS[floor.floor_id % len(FloorVisualizer3D.FLOOR_COLORS)]
            floor_mesh = FloorVisualizer3D.create_floor_slab_mesh(
                floor, color, xy_bounds
            )
            geometries.append(floor_mesh)
        
        # Add coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)
        
        # Add text labels for floors (as point clouds with colors)
        for floor in floors:
            # Create a small sphere at floor center as a marker
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate([0, 0, floor.z_center])
            color = FloorVisualizer3D.FLOOR_COLORS[floor.floor_id % len(FloorVisualizer3D.FLOOR_COLORS)]
            sphere.paint_uniform_color(color)
            geometries.append(sphere)
        
        # Visualize
        if interactive:
            print(f"\n{'='*70}")
            print(f"Interactive 3D Visualization: {scene_name}")
            print(f"{'='*70}")
            print(f"Floors: {len(floors)}")
            print(f"Regions: {len(df_regions)}")
            print(f"\nControls:")
            print(f"  - Mouse drag: Rotate view")
            print(f"  - Scroll: Zoom in/out")
            print(f"  - Shift + drag: Pan view")
            print(f"  - H: Show help")
            print(f"  - Q/ESC: Exit")
            print(f"{'='*70}\n")
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Floor Decomposition: {scene_name}", 
                            width=1600, height=1200)
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            # Set view point
            ctr = vis.get_view_control()
            ctr.set_zoom(0.5)
            ctr.set_front([0.5, 0.5, -0.7])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
            
            # Render options
            opt = vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])
            opt.point_size = 5.0
            opt.line_width = 3.0
            
            vis.run()
            
            if save_path:
                vis.capture_screen_image(save_path)
                print(f"Saved screenshot to: {save_path}")
            
            vis.destroy_window()
        
        elif save_path:
            # Non-interactive: just render and save
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1920, height=1080)
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            ctr = vis.get_view_control()
            ctr.set_zoom(0.5)
            ctr.set_front([0.5, 0.5, -0.7])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
            
            opt = vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])
            opt.line_width = 3.0
            
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path)
            vis.destroy_window()
            
            print(f"Saved 3D visualization to: {save_path}")
    
    @staticmethod
    def visualize_floor_split_comparison(scene_name: str, original_region_csv: str,
                                        floor_dirs: List[Path], floors: List[FloorSlab]):
        """
        Side-by-side comparison of original scene vs floor-split scenes
        
        Args:
            scene_name: Scene identifier
            original_region_csv: Path to original region CSV
            floor_dirs: List of floor output directories
            floors: List of FloorSlab objects
        """
        print(f"\n{'='*70}")
        print(f"3D Comparison Visualization: {scene_name}")
        print(f"{'='*70}")
        print(f"Original scene → {len(floors)} floor sub-scenes")
        print(f"\nPress any key to cycle through floors...")
        print(f"{'='*70}\n")
        
        # Load original data
        df_original = pd.read_csv(original_region_csv)
        
        for i, (floor, floor_dir) in enumerate(zip(floors, floor_dirs)):
            floor_name = f"{scene_name}_floor{floor.floor_id}"
            floor_region_csv = floor_dir / f"{floor_name}_region_result.csv"
            floor_object_csv = floor_dir / f"{floor_name}_object_result.csv"
            
            if not floor_region_csv.exists():
                print(f"Skipping floor {floor.floor_id}: CSV not found")
                continue
            
            print(f"\nShowing Floor {floor.floor_id} ({floor.z_min:.2f}m to {floor.z_max:.2f}m)")
            print(f"  Regions: {len(floor.region_ids)}")
            
            # Visualize this floor
            FloorVisualizer3D.visualize_scene_floors(
                floor_name,
                str(floor_region_csv),
                str(floor_object_csv),
                [floor],  # Just this floor
                interactive=True,
                save_path=None
            )
