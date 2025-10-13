"""
Scene parser - loads and processes region and object data from CSV files
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict


class SceneParser:
    """Step 1: Parse scene data from CSV files
    
    Loads region and object information from VLA-3D dataset CSVs.
    Filters out non-navigable objects (walls, ceilings, floors).
    """
    
    def __init__(self, region_csv: str, object_csv: str, scene_name: str):
        self.scene_name = scene_name
        self.regions = {}
        self.objects = {}
        self.region_objects = defaultdict(list)  # region_id -> list of object_ids
        
        self._load_regions(region_csv)
        self._load_objects(object_csv)
    
    def _load_regions(self, region_csv: str):
        """Load region data from CSV"""
        df = pd.read_csv(region_csv)
        
        for _, row in df.iterrows():
            region_id = str(row['region_id'])
            self.regions[region_id] = {
                'region_id': region_id,
                'label': row['region_label'],
                'center': np.array([
                    row['region_bbox_cx'],
                    row['region_bbox_cy'],
                    row['region_bbox_cz']
                ]),
                'size': np.array([
                    row['region_bbox_xlength'],
                    row['region_bbox_ylength'],
                    row['region_bbox_zlength']
                ]),
                'bbox_min': np.array([
                    row['region_bbox_cx'] - row['region_bbox_xlength'] / 2,
                    row['region_bbox_cy'] - row['region_bbox_ylength'] / 2,
                    row['region_bbox_cz'] - row['region_bbox_zlength'] / 2
                ]),
                'bbox_max': np.array([
                    row['region_bbox_cx'] + row['region_bbox_xlength'] / 2,
                    row['region_bbox_cy'] + row['region_bbox_ylength'] / 2,
                    row['region_bbox_cz'] + row['region_bbox_zlength'] / 2
                ])
            }
        
        print(f"✓ Loaded {len(self.regions)} regions")
    
    def _load_objects(self, object_csv: str):
        """Load object data from CSV, filtering navigable objects"""
        df = pd.read_csv(object_csv)
        
        # Objects to exclude from navigation
        non_navigable = ['wall', 'ceiling', 'floor', 'void', 'remove']
        
        for _, row in df.iterrows():
            # Skip invalid objects
            if row['nyu40_label'] in non_navigable:
                continue
            
            object_id = str(row['object_id'])
            region_id = str(row['region_id'])
            
            self.objects[object_id] = {
                'object_id': object_id,
                'region_id': region_id,
                'label': row['nyu40_label'],
                'center': np.array([
                    row['object_bbox_cx'],
                    row['object_bbox_cy'],
                    row['object_bbox_cz']
                ]),
                'size': np.array([
                    row['object_bbox_xlength'],
                    row['object_bbox_ylength'],
                    row['object_bbox_zlength']
                ])
            }
            
            # Track which objects are in which regions
            self.region_objects[region_id].append(object_id)
        
        print(f"✓ Loaded {len(self.objects)} navigable objects")
    
    def get_region(self, region_id: str) -> Optional[Dict]:
        """Get region by ID"""
        return self.regions.get(region_id)
    
    def get_object(self, object_id: str) -> Optional[Dict]:
        """Get object by ID"""
        return self.objects.get(object_id)
    
    def get_objects_in_region(self, region_id: str) -> List[str]:
        """Get list of object IDs in a region"""
        return self.region_objects.get(region_id, [])
    
    def get_node_info(self, node_id: str) -> Dict:
        """Get information about a node (region or object)"""
        if node_id.startswith('region_'):
            region_id = node_id.replace('region_', '')
            region = self.get_region(region_id)
            if region:
                return {
                    'type': 'region',
                    'id': node_id,
                    'label': region['label'],
                    'coordinates': region['center'],
                    'region_id': region_id
                }
        elif node_id.startswith('object_'):
            object_id = node_id.replace('object_', '')
            obj = self.get_object(object_id)
            if obj:
                return {
                    'type': 'object',
                    'id': node_id,
                    'label': obj['label'],
                    'coordinates': obj['center'],
                    'region_id': obj['region_id']
                }
        return None
