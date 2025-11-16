#!/usr/bin/env python3
"""
Enhance floor-decomposed datasets to be completely self-contained
Adds original_id mappings and all necessary metadata to each floor directory
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm
import shutil
import open3d as o3d


class FloorDatasetEnhancer:
    """Add complete metadata to make floor datasets self-contained"""
    
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
    
    def enhance_floor_metadata(self, floor_dir: Path, scene_name: str, floor_id: int):
        """
        Enhance floor metadata with complete mapping information
        """
        metadata_file = floor_dir / f"{scene_name}_floor{floor_id}_floor_metadata.json"
        
        if not metadata_file.exists():
            print(f"  Warning: Metadata not found: {metadata_file}")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Add reverse mappings (remapped_id -> original_id)
        if 'object_id_mapping' in metadata:
            reverse_obj_mapping = {
                int(v): int(k) 
                for k, v in metadata['object_id_mapping'].items()
            }
            metadata['remapped_to_original_object'] = reverse_obj_mapping
        
        if 'region_id_mapping' in metadata:
            reverse_reg_mapping = {
                int(v): int(k)
                for k, v in metadata['region_id_mapping'].items()
            }
            metadata['remapped_to_original_region'] = reverse_reg_mapping
        
        # Add dataset version info
        metadata['dataset_version'] = '1.0'
        metadata['self_contained'] = True
        metadata['requires_original_dataset'] = False
        
        # Save enhanced metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def add_object_lookup(self, floor_dir: Path, scene_name: str, floor_id: int):
        """
        Create a fast lookup JSON for objects with both IDs
        """
        object_csv = floor_dir / f"{scene_name}_floor{floor_id}_object_result.csv"
        metadata_file = floor_dir / f"{scene_name}_floor{floor_id}_floor_metadata.json"
        
        if not object_csv.exists() or not metadata_file.exists():
            return
        
        # Load metadata for mappings
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load objects
        df = pd.read_csv(object_csv)
        
        # Create lookup structure
        object_lookup = {}
        
        for idx, row in df.iterrows():
            remapped_id = int(row['object_id'])
            original_id = metadata['remapped_to_original_object'].get(remapped_id, remapped_id)
            
            # Extract color info
            color = row.get('object_color_scheme1', '_')
            color_rgb = None
            if pd.notna(row.get('object_color_r1')):
                color_rgb = [
                    int(row['object_color_r1']),
                    int(row['object_color_g1']),
                    int(row['object_color_b1'])
                ]
            
            # Extract bbox info
            bbox_dims = None
            if pd.notna(row.get('object_bbox_xlength')):
                bbox_dims = [
                    float(row['object_bbox_xlength']),
                    float(row['object_bbox_ylength']),
                    float(row['object_bbox_zlength'])
                ]
            
            # Position
            position = None
            if pd.notna(row.get('object_bbox_cx')):
                position = [
                    float(row['object_bbox_cx']),
                    float(row['object_bbox_cy']),
                    float(row['object_bbox_cz'])
                ]
            
            object_lookup[str(remapped_id)] = {
                'id': remapped_id,
                'original_id': original_id,
                'label': str(row.get('nyu40_label', row.get('raw_label', 'unknown'))),
                'raw_label': str(row.get('raw_label', 'unknown')),
                'nyu40_label': str(row.get('nyu40_label', 'unknown')),
                'color': str(color) if color != '_' else None,
                'color_rgb': color_rgb,
                'position': position,
                'bbox_dimensions': bbox_dims,
                'parent_region_id': int(row['region_id']),
                'heading': float(row.get('object_bbox_heading', 0.0)) if pd.notna(row.get('object_bbox_heading')) else None
            }
        
        # Save lookup
        lookup_file = floor_dir / f"{scene_name}_floor{floor_id}_object_lookup.json"
        with open(lookup_file, 'w') as f:
            json.dump(object_lookup, f, indent=2)
        
        print(f"  ✓ Created object lookup: {len(object_lookup)} objects")
        return object_lookup
    
    def add_region_lookup(self, floor_dir: Path, scene_name: str, floor_id: int):
        """
        Create a fast lookup JSON for regions with both IDs
        """
        region_csv = floor_dir / f"{scene_name}_floor{floor_id}_region_result.csv"
        metadata_file = floor_dir / f"{scene_name}_floor{floor_id}_floor_metadata.json"
        
        if not region_csv.exists() or not metadata_file.exists():
            return
        
        # Load metadata for mappings
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load regions
        df = pd.read_csv(region_csv)
        
        # Create lookup structure
        region_lookup = {}
        
        for idx, row in df.iterrows():
            remapped_id = int(row['region_id'])
            original_id = metadata['remapped_to_original_region'].get(remapped_id, remapped_id)
            
            # Position (region center)
            position = None
            if pd.notna(row.get('region_bbox_cx')):
                position = [
                    float(row['region_bbox_cx']),
                    float(row['region_bbox_cy']),
                    float(row['region_bbox_cz'])
                ]
            
            # Bbox dimensions
            bbox_dims = None
            if pd.notna(row.get('region_bbox_xlength')):
                bbox_dims = [
                    float(row['region_bbox_xlength']),
                    float(row['region_bbox_ylength']),
                    float(row['region_bbox_zlength'])
                ]
            
            region_lookup[str(remapped_id)] = {
                'id': remapped_id,
                'original_id': original_id,
                'label': str(row['region_label']),
                'position': position,
                'bbox_dimensions': bbox_dims
            }
        
        # Save lookup
        lookup_file = floor_dir / f"{scene_name}_floor{floor_id}_region_lookup.json"
        with open(lookup_file, 'w') as f:
            json.dump(region_lookup, f, indent=2)
        
        print(f"  ✓ Created region lookup: {len(region_lookup)} regions")
        return region_lookup
    
    def crop_and_save_ply(self, floor_dir: Path, scene_name: str, floor_id: int):
        """
        Crop PLY point cloud to floor-specific z-range and save
        """
        # Load floor metadata to get z-range
        metadata_file = floor_dir / f"{scene_name}_floor{floor_id}_floor_metadata.json"
        if not metadata_file.exists():
            print(f"  Warning: Metadata not found, cannot crop PLY")
            return False
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        z_min = metadata.get('z_min')
        z_max = metadata.get('z_max')
        
        if z_min is None or z_max is None:
            print(f"  Warning: z_min/z_max not in metadata, cannot crop PLY")
            return False
        
        # Path to original point cloud
        original_dataset_root = Path(__file__).parent.parent / 'VLA-3D_dataset' / 'Matterport'
        ply_source = original_dataset_root / scene_name / f"{scene_name}_pc_result.ply"
        
        if not ply_source.exists():
            print(f"  Warning: PLY file not found: {ply_source}")
            return False
        
        # Output path
        ply_dest = floor_dir / f"{scene_name}_floor{floor_id}.ply"
        
        if ply_dest.exists():
            print(f"  ℹ Floor PLY already exists: {ply_dest.name}")
            return True
        
        try:
            # Load original point cloud
            pcd = o3d.io.read_point_cloud(str(ply_source))
            
            if not pcd.has_points():
                print(f"  Warning: Empty point cloud")
                return False
            
            # Get points as numpy array
            points = np.asarray(pcd.points)
            
            # Filter by z-range (with small tolerance)
            z_tolerance = 0.5  # 0.5m buffer
            mask = (points[:, 2] >= (z_min - z_tolerance)) & (points[:, 2] <= (z_max + z_tolerance))
            
            # Create new point cloud with filtered points
            floor_pcd = o3d.geometry.PointCloud()
            floor_pcd.points = o3d.utility.Vector3dVector(points[mask])
            
            # Copy colors if available
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                floor_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            
            # Copy normals if available
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
                floor_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
            
            # Save cropped point cloud
            o3d.io.write_point_cloud(str(ply_dest), floor_pcd)
            
            file_size_mb = ply_dest.stat().st_size / (1024 * 1024)
            reduction = (1 - len(floor_pcd.points) / len(pcd.points)) * 100
            print(f"  ✓ Cropped PLY: {len(floor_pcd.points):,} points ({file_size_mb:.1f} MB, {reduction:.1f}% reduction)")
            return True
            
        except Exception as e:
            print(f"  Error cropping PLY: {e}")
            return False
    
    def enhance_floor(self, scene_dir: Path):
        """
        Enhance all floors in a scene directory
        """
        scene_name = scene_dir.name
        floor_dirs = sorted(scene_dir.glob('floor_*'))
        
        if not floor_dirs:
            return
        
        print(f"\nEnhancing scene: {scene_name} ({len(floor_dirs)} floors)")
        
        for floor_dir in floor_dirs:
            floor_id = int(floor_dir.name.split('_')[1])
            print(f"  Floor {floor_id}:")
            
            # Enhance metadata with reverse mappings
            self.enhance_floor_metadata(floor_dir, scene_name, floor_id)
            
            # Create object lookup
            self.add_object_lookup(floor_dir, scene_name, floor_id)
            
            # Create region lookup
            self.add_region_lookup(floor_dir, scene_name, floor_id)
            
            # Crop and save floor-specific PLY
            self.crop_and_save_ply(floor_dir, scene_name, floor_id)
    
    def enhance_all_floors(self):
        """
        Enhance all floor datasets in the output directory
        """
        scene_dirs = sorted([d for d in self.output_root.iterdir() if d.is_dir()])
        
        print(f"Found {len(scene_dirs)} scenes to enhance")
        
        for scene_dir in tqdm(scene_dirs, desc="Enhancing scenes"):
            try:
                self.enhance_floor(scene_dir)
            except Exception as e:
                print(f"Error enhancing {scene_dir.name}: {e}")
        
        print("\n✓ All floors enhanced and self-contained!")


def main():
    parser = argparse.ArgumentParser(
        description='Enhance floor datasets to be completely self-contained'
    )
    parser.add_argument('--output_root', type=str, required=True,
                       help='Path to floor-decomposed output (e.g., output/floors)')
    
    args = parser.parse_args()
    
    enhancer = FloorDatasetEnhancer(args.output_root)
    enhancer.enhance_all_floors()


if __name__ == '__main__':
    main()
