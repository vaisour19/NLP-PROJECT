# 3D Language-grounded Navigation Dataset

A comprehensive dataset for vision-language-action research featuring floor-decomposed 3D scenes with natural language navigation tasks.

## Overview

This dataset provides navigation tasks in 3D indoor environments from the Matterport3D dataset. Each scene is decomposed into individual floors with cropped point clouds, object/region metadata, and thousands of natural language navigation instructions.

### Dataset Statistics

- **90 scenes** from Matterport3D
- **183 floors** across all scenes
- **182,000+ navigation tasks** (1000 tasks per floor)
- **13GB total size** (optimized with floor-specific point clouds)

### Key Features

- Floor-level decomposition of 3D scenes
- Cropped point clouds per floor (40-90% size reduction)
- Rich spatial metadata (objects, regions, relationships)
- Natural language navigation instructions with multiple variants
- Difficulty-based task classification (easy, medium, hard)
- Object and region grounding with 3D bounding boxes

## Dataset Structure

```
output/floors/
├── [scene_name]/
│   ├── floor_0/
│   │   ├── [scene]_floor0.ply                          # Cropped point cloud
│   │   ├── [scene]_floor0_object_lookup.json           # Object metadata
│   │   ├── [scene]_floor0_region_lookup.json           # Region metadata  
│   │   ├── [scene]_floor0_floor_metadata.json          # Floor information
│   │   ├── [scene]_floor0_scene_graph.json             # Scene graph
│   │   ├── [scene]_floor0_referential_statements.json  # Language grounding
│   │   ├── [scene]_floor0_object_result.csv            # Object CSV
│   │   ├── [scene]_floor0_region_result.csv            # Region CSV
│   │   └── navigation_tasks/
│   │       └── [scene]_floor0_tasks.jsonl              # 1000 navigation tasks
│   ├── floor_1/
│   └── floor_2/
```

## Navigation Task Format

Each task is a JSON object containing:

```json
{
  "task_id": "scene_floor0_task_0",
  "scene_name": "1LXtFkjw3qL",
  "floor_id": 0,
  "instruction": "Navigate from the white chair to the wooden table",
  "instruction_variants": [...],
  "start_point": {
    "type": "object",
    "id": "191",
    "description": "white upholstered chair",
    "position": [x, y, z],
    "alternative_descriptions": [...]
  },
  "end_point": {
    "type": "object", 
    "id": "156",
    "description": "wooden dining table",
    "position": [x, y, z],
    "alternative_descriptions": [...]
  },
  "spatial_metadata": {
    "euclidean_distance": 4.52,
    "difficulty": "medium",
    "floor_id": 0
  }
}
```

## Installation

### Prerequisites

- Python 3.10+
- micromamba or conda

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd 3d-ln

# Create environment
micromamba create -n floor-decomp python=3.10
micromamba activate floor-decomp

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

- open3d >= 0.18.0
- numpy
- pandas
- tqdm
- scipy

## Usage

### Dataset Generation

#### 1. Floor Decomposition

Decompose scenes into individual floors with cropped point clouds:

```bash
python enhance_floor_datasets.py --output_root output/floors
```

This creates:
- Per-floor point clouds (cropped to floor z-range)
- Object and region lookups per floor
- Floor metadata with height information

#### 2. Navigation Task Generation

Generate natural language navigation tasks:

```bash
python generate_navigation_tasks.py --output_root output/floors --num_tasks 1000
```

Options:
- `--output_root`: Directory containing floor-decomposed data
- `--num_tasks`: Number of tasks to generate per floor (default: 1000)

### Visualization

Interactive 3D visualization with GUI controls:

```bash
python visualizer.py --dataset_dir output/floors
```

Features:
- Scene and floor selection dropdowns
- Task navigation (previous/next)
- Point cloud size adjustment
- Visibility toggles (objects, regions, point cloud)
- Label display (object names, region names)
- High-contrast color coding (magenta for start, green for end)

## Navigation Task Types

Tasks are categorized by difficulty based on spatial complexity:

### Easy Tasks
- Direct line-of-sight navigation
- Short distances (< 3m)
- Minimal obstacles

### Medium Tasks
- Moderate distances (3-7m)
- Some spatial reasoning required
- Multiple reference points

### Hard Tasks
- Long distances (> 7m)
- Complex spatial relationships
- Multi-step reasoning required

## Object and Region Metadata

### Object Attributes
- Unique ID
- Category label (NYU40 classes)
- 3D position (x, y, z)
- Bounding box dimensions
- Heading/rotation
- Color information

### Region Attributes
- Region type (room, hallway, etc.)
- 3D bounding box
- Floor assignment
- Connected objects
- Spatial relationships

## Spatial Relationships

The dataset includes rich spatial relationship annotations:
- **Directional**: above, below, in front of, behind
- **Proximity**: near, closest, farthest
- **Containment**: in, on
- **Relative ordering**: second closest, third farthest

## Data Format Specifications

### Point Cloud (.ply)
- Format: ASCII PLY
- Properties: x, y, z, red, green, blue
- Coordinate system: Right-handed Z-up

### Object Lookup (JSON)
```json
{
  "object_id": {
    "label": "chair",
    "position": [x, y, z],
    "bbox_dimensions": [w, h, d],
    "heading": 0.0,
    "floor_id": 0
  }
}
```

### Navigation Tasks (JSONL)
One task per line in JSON format, enabling efficient streaming and processing.

## Quality Assurance

- All navigation tasks verified for valid start/end points
- Distance calculations validated
- Spatial relationships grounded in 3D geometry
- Multiple instruction variants per task for diversity

## Acknowledgments

- Matterport3D dataset for providing the base 3D environments
- Open3D library for 3D visualization and processing

## Contact

For questions or issues, please open an issue in the repository.

## Version History

### v1.0.0 (November 2025)
- Initial release
- 90 scenes, 183 floors
- 182,000+ navigation tasks
- Floor-decomposed dataset with optimized storage
