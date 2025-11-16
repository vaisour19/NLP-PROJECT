"""
Split scene data into per-floor sub-datasets
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set

try:
    from .floor_detector import FloorSlab
except ImportError:
    from floor_detector import FloorSlab


class DatasetSplitter:
    """Split multi-floor scene into single-floor sub-scenes"""
    
    def __init__(self, scene_name: str, scene_dir: Path, output_root: Path):
        """
        Args:
            scene_name: Scene identifier (e.g., "1LXtFkjw3qL")
            scene_dir: Path to original scene directory
            output_root: Root directory for output floor-split data
        """
        self.scene_name = scene_name
        self.scene_dir = Path(scene_dir)
        self.output_root = Path(output_root)
        
        # Expected file paths
        self.region_csv = self.scene_dir / f"{scene_name}_region_result.csv"
        self.object_csv = self.scene_dir / f"{scene_name}_object_result.csv"
        self.scene_graph_json = self.scene_dir / f"{scene_name}_scene_graph.json"
        self.ref_statements_json = self.scene_dir / f"{scene_name}_referential_statements.json"
    
    def split_regions_csv(self, floor: FloorSlab) -> pd.DataFrame:
        """Filter and remap region CSV for a floor"""
        df = pd.read_csv(self.region_csv)
        
        # Filter to regions in this floor
        floor_regions = df[df['region_id'].isin(floor.region_ids)].copy()
        
        # Create old_id -> new_id mapping
        old_ids = sorted(floor.region_ids)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        
        # Remap region_ids
        floor_regions['region_id'] = floor_regions['region_id'].map(id_mapping)
        
        # Sort by new region_id
        floor_regions = floor_regions.sort_values('region_id').reset_index(drop=True)
        
        return floor_regions, id_mapping
    
    def split_objects_csv(self, floor: FloorSlab, region_id_mapping: Dict[int, int]) -> pd.DataFrame:
        """Filter and remap object CSV for a floor"""
        df = pd.read_csv(self.object_csv)
        
        # Filter objects belonging to regions in this floor
        floor_objects = df[df['region_id'].isin(region_id_mapping.keys())].copy()
        
        if len(floor_objects) == 0:
            return floor_objects, {}
        
        # Additionally filter by Z-coordinate overlap
        filtered_objects = []
        for idx, row in floor_objects.iterrows():
            obj_z = row['object_bbox_cz']
            obj_z_len = row['object_bbox_zlength']
            obj_z_min = obj_z - obj_z_len / 2
            obj_z_max = obj_z + obj_z_len / 2
            
            # Check if object overlaps with floor Z range
            if floor.overlaps(obj_z_min, obj_z_max, tolerance=0.5):
                filtered_objects.append(row)
        
        if not filtered_objects:
            return pd.DataFrame(), {}
        
        floor_objects = pd.DataFrame(filtered_objects).reset_index(drop=True)
        
        # Remap region_ids
        floor_objects['region_id'] = floor_objects['region_id'].map(region_id_mapping)
        
        # Create object_id mapping
        old_obj_ids = floor_objects['object_id'].tolist()
        obj_id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_obj_ids)}
        
        # Remap object_ids
        floor_objects['object_id'] = floor_objects['object_id'].map(obj_id_mapping)
        
        # Sort by new object_id
        floor_objects = floor_objects.sort_values('object_id').reset_index(drop=True)
        
        return floor_objects, obj_id_mapping
    
    def split_scene_graph(self, floor: FloorSlab, region_id_mapping: Dict[int, int], 
                          obj_id_mapping: Dict[int, int]) -> Dict:
        """Filter and remap scene graph JSON for a floor"""
        with open(self.scene_graph_json, 'r') as f:
            scene_graph = json.load(f)
        
        floor_scene_graph = {
            'scene_name': f"{self.scene_name}_floor{floor.floor_id}",
            'regions': {}
        }
        
        # Filter regions
        for old_region_id_str, region_data in scene_graph['regions'].items():
            old_region_id = int(old_region_id_str)
            
            if old_region_id not in region_id_mapping:
                continue
            
            new_region_id = region_id_mapping[old_region_id]
            new_region_data = region_data.copy()
            new_region_data['region_id'] = str(new_region_id)
            
            # Remap object IDs in objects list
            if 'objects' in new_region_data:
                new_objects = []
                for obj in new_region_data['objects']:
                    if 'object_id' in obj:
                        old_obj_id = int(obj['object_id'])
                        if old_obj_id in obj_id_mapping:
                            obj_copy = obj.copy()
                            obj_copy['object_id'] = str(obj_id_mapping[old_obj_id])
                            new_objects.append(obj_copy)
                new_region_data['objects'] = new_objects
            
            # Remap relationships (if present)
            if 'relationships' in new_region_data:
                # Keep relationships but remap object references
                # This is complex - for now we'll keep simplified version
                new_region_data['relationships'] = {}
            
            floor_scene_graph['regions'][str(new_region_id)] = new_region_data
        
        return floor_scene_graph
    
    def split_referential_statements(self, floor: FloorSlab, region_id_mapping: Dict[int, int]) -> Dict:
        """Filter and remap referential statements JSON for a floor"""
        if not self.ref_statements_json.exists():
            return {
                'scene_name': f"{self.scene_name}_floor{floor.floor_id}",
                'regions': {}
            }
        
        with open(self.ref_statements_json, 'r') as f:
            ref_statements = json.load(f)
        
        floor_ref_statements = {
            'scene_name': f"{self.scene_name}_floor{floor.floor_id}",
            'regions': {}
        }
        
        # Filter regions
        for old_region_id_str, statements in ref_statements['regions'].items():
            old_region_id = int(old_region_id_str)
            
            if old_region_id not in region_id_mapping:
                continue
            
            new_region_id = region_id_mapping[old_region_id]
            floor_ref_statements['regions'][str(new_region_id)] = statements
        
        return floor_ref_statements
    
    def split_floor(self, floor: FloorSlab) -> Path:
        """
        Split all data files for a single floor
        
        Returns:
            Path to output floor directory
        """
        # Create output directory
        floor_dir = self.output_root / self.scene_name / f"floor_{floor.floor_id}"
        floor_dir.mkdir(parents=True, exist_ok=True)
        
        floor_name = f"{self.scene_name}_floor{floor.floor_id}"
        
        # Split regions CSV
        floor_regions, region_id_mapping = self.split_regions_csv(floor)
        region_csv_path = floor_dir / f"{floor_name}_region_result.csv"
        floor_regions.to_csv(region_csv_path, index=False)
        
        # Split objects CSV
        floor_objects, obj_id_mapping = self.split_objects_csv(floor, region_id_mapping)
        object_csv_path = floor_dir / f"{floor_name}_object_result.csv"
        floor_objects.to_csv(object_csv_path, index=False)
        
        # Split scene graph
        if self.scene_graph_json.exists():
            floor_scene_graph = self.split_scene_graph(floor, region_id_mapping, obj_id_mapping)
            scene_graph_path = floor_dir / f"{floor_name}_scene_graph.json"
            with open(scene_graph_path, 'w') as f:
                json.dump(floor_scene_graph, f, indent=2)
        
        # Split referential statements
        if self.ref_statements_json.exists():
            floor_ref_statements = self.split_referential_statements(floor, region_id_mapping)
            ref_path = floor_dir / f"{floor_name}_referential_statements.json"
            with open(ref_path, 'w') as f:
                json.dump(floor_ref_statements, f, indent=2)
        
        # Write floor metadata
        metadata = {
            'scene_name': self.scene_name,
            'floor_id': floor.floor_id,
            'z_min': floor.z_min,
            'z_max': floor.z_max,
            'z_center': floor.z_center,
            'z_range': floor.z_max - floor.z_min,
            'num_regions': len(floor.region_ids),
            'num_objects': len(obj_id_mapping),
            'original_region_ids': floor.region_ids,
            'region_id_mapping': {str(k): v for k, v in region_id_mapping.items()},
            'object_id_mapping': {str(k): v for k, v in obj_id_mapping.items()}
        }
        
        metadata_path = floor_dir / f"{floor_name}_floor_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return floor_dir
