"""
Detect and cluster floor levels from region bounding boxes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class FloorSlab:
    """Represents a single floor level"""
    floor_id: int
    z_min: float
    z_max: float
    z_center: float
    region_ids: List[int]
    
    def overlaps(self, z_min: float, z_max: float, tolerance: float = 0.4) -> bool:
        """Check if this floor overlaps with given Z range"""
        return not (z_max < self.z_min - tolerance or z_min > self.z_max + tolerance)
    
    def merge(self, z_min: float, z_max: float, region_id: int):
        """Merge a new region into this floor"""
        self.z_min = min(self.z_min, z_min)
        self.z_max = max(self.z_max, z_max)
        self.z_center = (self.z_min + self.z_max) / 2
        self.region_ids.append(region_id)


class FloorDetector:
    """Detect floor levels from region CSV data"""
    
    def __init__(self, z_tolerance: float = 0.4, exclude_stairs: bool = True):
        """
        Args:
            z_tolerance: Tolerance for merging overlapping floors (meters)
            exclude_stairs: Whether to exclude stair regions from floor assignment
        """
        self.z_tolerance = z_tolerance
        self.exclude_stairs = exclude_stairs
        self.stair_keywords = ['stairs', 'stairway', 'staircase', 'elevator']
    
    def is_stairs(self, region_label: str) -> bool:
        """Check if region is a stairway/elevator"""
        label_lower = region_label.lower()
        return any(keyword in label_lower for keyword in self.stair_keywords)
    
    def detect_floors(self, region_csv_path: str) -> List[FloorSlab]:
        """
        Detect floor levels from region CSV using Z-center clustering
        
        Args:
            region_csv_path: Path to *_region_result.csv
            
        Returns:
            List of FloorSlab objects, sorted by z_center
        """
        # Load regions
        df = pd.read_csv(region_csv_path)
        
        if len(df) == 0:
            return []
        
        # Calculate Z bounds for each region
        regions_data = []
        for idx, row in df.iterrows():
            region_id = int(row['region_id'])
            region_label = row['region_label']
            z_center = row['region_bbox_cz']
            z_length = row['region_bbox_zlength']
            
            z_min = z_center - z_length / 2
            z_max = z_center + z_length / 2
            
            # Skip stairs if configured
            if self.exclude_stairs and self.is_stairs(region_label):
                continue
            
            regions_data.append({
                'region_id': region_id,
                'region_label': region_label,
                'z_min': z_min,
                'z_max': z_max,
                'z_center': z_center
            })
        
        if not regions_data:
            return []
        
        # Sort by z_center (not z_min, to avoid issues with tall rooms)
        regions_data.sort(key=lambda x: x['z_center'])
        
        # Cluster based on Z-center gaps (improved algorithm)
        floors: List[FloorSlab] = []
        current_floor_regions = [regions_data[0]]
        
        for i in range(1, len(regions_data)):
            prev_region = regions_data[i-1]
            curr_region = regions_data[i]
            
            # Check gap between Z-centers
            z_gap = curr_region['z_center'] - prev_region['z_center']
            
            # If gap is large, start a new floor
            # Use a threshold of 1.5m (typical floor height difference)
            if z_gap > 1.5:
                # Finalize current floor
                self._create_floor_from_regions(floors, current_floor_regions)
                current_floor_regions = [curr_region]
            else:
                current_floor_regions.append(curr_region)
        
        # Don't forget the last floor
        if current_floor_regions:
            self._create_floor_from_regions(floors, current_floor_regions)
        
        # Reassign floor_ids after sorting
        for i, floor in enumerate(floors):
            floor.floor_id = i
        
        return floors
    
    def _create_floor_from_regions(self, floors: List[FloorSlab], regions: List[dict]):
        """Helper to create a FloorSlab from a list of regions"""
        if not regions:
            return
        
        region_ids = [r['region_id'] for r in regions]
        z_mins = [r['z_min'] for r in regions]
        z_maxs = [r['z_max'] for r in regions]
        z_centers = [r['z_center'] for r in regions]
        
        floor = FloorSlab(
            floor_id=len(floors),
            z_min=min(z_mins),
            z_max=max(z_maxs),
            z_center=np.mean(z_centers),
            region_ids=region_ids
        )
        
        floors.append(floor)
    
    def validate_floors(self, floors: List[FloorSlab]) -> Dict[str, any]:
        """
        Validate detected floors
        
        Returns:
            Dictionary with validation statistics
        """
        if not floors:
            return {
                'valid': False,
                'num_floors': 0,
                'error': 'No floors detected'
            }
        
        stats = {
            'valid': True,
            'num_floors': len(floors),
            'floor_heights': [],
            'regions_per_floor': []
        }
        
        for floor in floors:
            height = floor.z_max - floor.z_min
            stats['floor_heights'].append(height)
            stats['regions_per_floor'].append(len(floor.region_ids))
        
        stats['avg_height'] = np.mean(stats['floor_heights'])
        stats['total_regions'] = sum(stats['regions_per_floor'])
        
        return stats
