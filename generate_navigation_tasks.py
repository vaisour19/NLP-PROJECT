#!/usr/bin/env python3
"""
Optimized Navigation Task Generator
Generates 1000 navigation tasks per floor with natural language instructions
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm


@dataclass
class Point:
    """Represents a navigation waypoint (object or region)"""
    type: str  # "object" or "region"
    id: int
    label: str
    color: Optional[str]
    color_rgb: Optional[List[int]]
    position: List[float]
    bbox_dimensions: Optional[List[float]]
    parent_region_id: Optional[int]
    parent_region_name: Optional[str]


class NavigationTaskGenerator:
    """Fast, optimized navigation task generator"""
    
    def __init__(self, floor_dir: Path, scene_name: str, floor_id: int):
        self.floor_dir = Path(floor_dir)
        self.scene_name = scene_name
        self.floor_id = floor_id
        
        # Load all data
        self.objects = self._load_json(f'{scene_name}_floor{floor_id}_object_lookup.json')
        self.regions = self._load_json(f'{scene_name}_floor{floor_id}_region_lookup.json')
        self.templates = self._load_templates()
        
        # Cache for performance
        self.valid_objects = [obj for obj in self.objects.values() if obj.get('position')]
        self.valid_regions = list(self.regions.values())
        
        # Validate floor has sufficient data
        if len(self.valid_objects) == 0 and len(self.valid_regions) < 2:
            raise ValueError(f"Insufficient data: {len(self.valid_objects)} objects, {len(self.valid_regions)} regions")
        
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file"""
        with open(self.floor_dir / filename) as f:
            return json.load(f)
    
    def _load_templates(self) -> Dict:
        """Load natural language templates"""
        template_file = Path(__file__).parent / 'natural_language_templates.json'
        with open(template_file) as f:
            return json.load(f)
    
    def _euclidean_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate 3D Euclidean distance"""
        return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
    
    def _get_region_name(self, region_id: int) -> str:
        """Get region name by ID"""
        return self.regions.get(str(region_id), {}).get('label', 'unknown')
    
    def _create_point(self, data: Dict, point_type: str) -> Point:
        """Convert raw data to Point object"""
        if point_type == "object":
            return Point(
                type="object",
                id=data['id'],
                label=data['label'],
                color=data.get('color'),
                color_rgb=data.get('color_rgb'),
                position=data['position'],
                bbox_dimensions=data.get('bbox_dimensions'),
                parent_region_id=data.get('parent_region_id'),
                parent_region_name=self._get_region_name(data.get('parent_region_id'))
            )
        else:  # region
            return Point(
                type="region",
                id=data['id'],
                label=data['label'],
                color=None,
                color_rgb=None,
                position=data['position'],
                bbox_dimensions=data.get('bbox_dimensions'),
                parent_region_id=None,
                parent_region_name=None
            )
    
    def _generate_description(self, point: Point) -> str:
        """Generate primary description for a point"""
        if point.type == "object":
            if point.color and point.parent_region_name:
                return f"the {point.color} {point.label} in the {point.parent_region_name}"
            elif point.color:
                return f"the {point.color} {point.label}"
            elif point.parent_region_name:
                return f"the {point.label} in the {point.parent_region_name}"
            else:
                return f"the {point.label}"
        else:  # region
            return f"the {point.label}"
    
    def _generate_alternative_descriptions(self, point: Point) -> List[str]:
        """Generate 20 alternative descriptions"""
        alts = []
        patterns = self.templates['alternative_description_patterns']
        
        # Prepare substitutions
        subs = {
            '{label}': point.label,
            '{color}': point.color or '',
            '{region}': point.parent_region_name or ''
        }
        
        # Generate from patterns
        for pattern in patterns[:20]:
            desc = pattern
            for key, val in subs.items():
                desc = desc.replace(key, val)
            desc = desc.strip()
            desc = ' '.join(desc.split())  # Remove extra spaces
            if desc and desc not in alts:
                alts.append(desc)
        
        # Ensure exactly 20
        while len(alts) < 20:
            alts.append(self._generate_description(point))
        
        return alts[:20]
    
    def _generate_instruction(self, start: Point, end: Point) -> str:
        """Generate main instruction"""
        start_desc = self._generate_description(start)
        end_desc = self._generate_description(end)
        return f"Walk from {start_desc} to {end_desc}"
    
    def _generate_instruction_variants(self, start: Point, end: Point) -> List[str]:
        """Generate 25 instruction variants using templates"""
        variants = []
        
        # Prepare substitutions
        start_desc = self._generate_description(start)
        end_desc = self._generate_description(end)
        
        subs = {
            '{start}': start_desc,
            '{end}': end_desc,
            '{start_label}': start.label,
            '{end_label}': end.label,
            '{start_color}': start.color or '',
            '{end_color}': end.color or '',
            '{start_region}': start.parent_region_name or '',
            '{end_region}': end.parent_region_name or ''
        }
        
        # Sample templates from different categories
        template_categories = self.templates['instruction_templates']
        
        for category_templates in template_categories.values():
            for template in category_templates[:3]:  # 3 per category
                variant = template
                for key, val in subs.items():
                    variant = variant.replace(key, val)
                variant = ' '.join(variant.split())  # Clean spaces
                if variant and variant not in variants:
                    variants.append(variant)
                if len(variants) >= 25:
                    break
            if len(variants) >= 25:
                break
        
        # Ensure exactly 25
        while len(variants) < 25:
            variants.append(f"Navigate from {start_desc} to {end_desc}")
        
        return variants[:25]
    
    def _select_point_pair(self, min_distance: float = 2.0) -> Tuple[Point, Point]:
        """Randomly select valid start and end points"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Randomly choose types (80% object-object, 10% object-region, 10% region-object)
            rand = random.random()
            if rand < 0.8:  # object to object
                start_data = random.choice(self.valid_objects)
                end_data = random.choice(self.valid_objects)
                start = self._create_point(start_data, "object")
                end = self._create_point(end_data, "object")
            elif rand < 0.9:  # object to region
                start_data = random.choice(self.valid_objects)
                end_data = random.choice(self.valid_regions)
                start = self._create_point(start_data, "object")
                end = self._create_point(end_data, "region")
            else:  # region to object
                start_data = random.choice(self.valid_regions)
                end_data = random.choice(self.valid_objects)
                start = self._create_point(start_data, "region")
                end = self._create_point(end_data, "object")
            
            # Validate
            if start.id != end.id:
                distance = self._euclidean_distance(start.position, end.position)
                if distance >= min_distance:
                    return start, end
        
        # Fallback: return any valid pair
        start_data = random.choice(self.valid_objects)
        end_data = random.choice(self.valid_objects)
        return (self._create_point(start_data, "object"),
                self._create_point(end_data, "object"))
    
    def _calculate_difficulty(self, distance: float, crosses_regions: bool) -> str:
        """Determine task difficulty"""
        if not crosses_regions and distance < 5:
            return "easy"
        elif distance > 15 or crosses_regions:
            return "hard"
        else:
            return "medium"
    
    def generate_task(self, task_num: int) -> Dict:
        """Generate a single navigation task"""
        # Select points
        start, end = self._select_point_pair()
        
        # Calculate spatial metadata
        distance = self._euclidean_distance(start.position, end.position)
        crosses_regions = (start.parent_region_id != end.parent_region_id 
                          if start.parent_region_id and end.parent_region_id else False)
        difficulty = self._calculate_difficulty(distance, crosses_regions)
        
        # Generate language
        instruction = self._generate_instruction(start, end)
        variants = self._generate_instruction_variants(start, end)
        start_alts = self._generate_alternative_descriptions(start)
        end_alts = self._generate_alternative_descriptions(end)
        
        # Build task
        task = {
            "task_id": f"{self.scene_name}_floor{self.floor_id}_task_{task_num:04d}",
            "scene_name": self.scene_name,
            "floor_id": self.floor_id,
            "instruction": instruction,
            "instruction_variants": variants,
            "spatial_metadata": {
                "euclidean_distance": round(distance, 2),
                "crosses_regions": crosses_regions,
                "start_region": start.parent_region_name,
                "end_region": end.parent_region_name,
                "difficulty": difficulty
            },
            "start_point": {
                "type": start.type,
                "id": start.id,
                "label": start.label,
                "color": start.color,
                "color_rgb": start.color_rgb,
                "position": [round(p, 2) for p in start.position],
                "bbox_dimensions": [round(d, 2) for d in start.bbox_dimensions] if start.bbox_dimensions else None,
                "parent_region_id": start.parent_region_id,
                "parent_region_name": start.parent_region_name,
                "description": self._generate_description(start),
                "alternative_descriptions": start_alts
            },
            "end_point": {
                "type": end.type,
                "id": end.id,
                "label": end.label,
                "color": end.color,
                "color_rgb": end.color_rgb,
                "position": [round(p, 2) for p in end.position],
                "bbox_dimensions": [round(d, 2) for d in end.bbox_dimensions] if end.bbox_dimensions else None,
                "parent_region_id": end.parent_region_id,
                "parent_region_name": end.parent_region_name,
                "description": self._generate_description(end),
                "alternative_descriptions": end_alts
            }
        }
        
        return task
    
    def generate_tasks(self, num_tasks: int = 1000) -> List[Dict]:
        """Generate multiple navigation tasks"""
        tasks = []
        for i in range(num_tasks):
            task = self.generate_task(i + 1)
            tasks.append(task)
        return tasks
    
    def save_tasks(self, tasks: List[Dict], output_file: Path):
        """Save tasks to JSONL file"""
        with open(output_file, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')


def generate_for_floor(floor_dir: Path, num_tasks: int = 1000) -> Dict:
    """Generate tasks for a single floor"""
    # Parse scene name and floor ID
    parts = floor_dir.name.split('_')
    floor_id = int(parts[-1])
    scene_name = floor_dir.parent.name
    
    try:
        # Initialize generator
        generator = NavigationTaskGenerator(floor_dir, scene_name, floor_id)
        
        # Generate tasks
        tasks = generator.generate_tasks(num_tasks)
        
        # Save tasks
        output_dir = floor_dir / 'navigation_tasks'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'{scene_name}_floor{floor_id}_tasks.jsonl'
        generator.save_tasks(tasks, output_file)
        
        return {
            'success': True,
            'floor': f'{scene_name}/floor_{floor_id}',
            'num_tasks': len(tasks),
            'output_file': str(output_file)
        }
        
    except Exception as e:
        return {
            'success': False,
            'floor': f'{scene_name}/floor_{floor_id}',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Generate navigation tasks for floor-decomposed dataset'
    )
    parser.add_argument('--output_root', type=str, default='output/floors',
                       help='Root directory of floor datasets')
    parser.add_argument('--num_tasks', type=int, default=1000,
                       help='Number of tasks per floor')
    parser.add_argument('--scenes', type=str, nargs='*',
                       help='Specific scenes to process (default: all)')
    
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    
    # Find all floor directories
    if args.scenes:
        floor_dirs = []
        for scene in args.scenes:
            scene_dir = output_root / scene
            if scene_dir.exists():
                floor_dirs.extend(sorted(scene_dir.glob('floor_*')))
    else:
        floor_dirs = sorted(output_root.glob('*/floor_*'))
    
    print(f"Found {len(floor_dirs)} floors to process")
    print(f"Generating {args.num_tasks} tasks per floor\n")
    
    # Process all floors
    results = []
    for floor_dir in tqdm(floor_dirs, desc="Generating tasks"):
        result = generate_for_floor(floor_dir, args.num_tasks)
        results.append(result)
    
    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total floors: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_tasks = sum(r['num_tasks'] for r in successful)
        print(f"\nTotal tasks generated: {total_tasks:,}")
        print(f"Average per floor: {total_tasks // len(successful):,}")
    
    if failed:
        print(f"\nFailed floors:")
        for r in failed:
            print(f"  {r['floor']}: {r['error']}")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
