# 🧭 VLA-3D Navigation Data Generator

**Allocentric (Absolute Direction) Action Space**

Generate synthetic navigation training data with compass direction actions for Vision-and-Language Navigation (VLN) tasks.

---

## 🎯 Action Space

```python
actions = ['north', 'south', 'east', 'west', 'up', 'down', 'stop']
```

**Coordinate System:**
- X-axis: East (+) / West (-)
- Y-axis: North (+) / South (-)
- Z-axis: Up (+) / Down (-)

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Generate Data for Single Scene

```bash
python3 generate_allocentric_data.py \
  --scene_dir ../dataset/Matterport/1LXtFkjw3qL \
  --scene_name 1LXtFkjw3qL \
  --num_samples 100 \
  --output_dir ./output_data \
  --step_size 1.0
```

### Merge Multiple Datasets

```bash
python3 merge_allocentric_datasets.py \
  --input_dir ./output_data \
  --output merged_dataset.json \
  --split
```

This creates train/val/test splits.

---

## 📁 Project Structure

```
navigation_generator/
├── src/                              # Core modules
│   ├── __init__.py                   # Package exports
│   ├── data_types.py                 # NavigationSample dataclass
│   ├── scene_parser.py               # CSV file parsing
│   ├── graph_builder.py              # Navigation graph construction
│   ├── pathfinder.py                 # A* pathfinding
│   ├── action_converter.py           # Path → Actions (ALLOCENTRIC)
│   ├── instruction_generator.py      # Natural language generation
│   └── data_generator.py             # Main pipeline orchestrator
│
├── generate_allocentric_data.py      # Main entry point
├── merge_allocentric_datasets.py     # Dataset merging utility
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

---

## 🔬 Pipeline Overview

The generator implements a 5-step pipeline:

1. **Parse Scene Data**: Load region/object CSVs from VLA-3D dataset
2. **Build Navigation Graph**: Create NetworkX graph with regions and objects as nodes
3. **Find Paths**: Use A* search to find optimal paths between random start/goal pairs
4. **Convert to Actions**: Transform 3D coordinates into allocentric action sequences
5. **Generate Instructions**: Create natural language descriptions of navigation tasks

---

## 📊 Output Format

```json
{
  "scene_name": "1LXtFkjw3qL",
  "num_samples": 100,
  "action_space": "allocentric",
  "actions": ["north", "south", "east", "west", "up", "down", "stop"],
  "samples": [
    {
      "task_id": "1LXtFkjw3qL_synthetic_0000",
      "instruction": "Go from the bedroom to the kitchen.",
      "action_sequence": ["north", "north", "east", "up", "stop"],
      "path_coordinates": [[0,0,0], [0,2,0], [1,2,0], [1,2,1]],
      "path_length": 4.0,
      "num_actions": 5,
      "has_vertical_movement": true
    }
  ]
}
```

---

## 🎛️ Command-Line Options

### generate_allocentric_data.py

```
--scene_dir PATH          Scene directory with CSV files (required)
--scene_name NAME         Scene ID (required)
--num_samples N           Number of samples (default: 100)
--output_dir PATH         Output directory (default: ./allocentric_data)
--step_size METERS        Distance per action (default: 1.0)
--seed N                  Random seed (default: 42)
--show_examples N         Examples to print (default: 5)
```

### merge_allocentric_datasets.py

```
--input_dir PATH          Directory with JSON files (required)
--output PATH             Output file path (required)
--split                   Split into train/val/test sets
--train_ratio FLOAT       Train set ratio (default: 0.8)
--val_ratio FLOAT         Validation set ratio (default: 0.1)
--test_ratio FLOAT        Test set ratio (default: 0.1)
```

---

## 💡 Example Workflow

### Process Multiple Scenes

```bash
for scene in 1LXtFkjw3qL 17DRP5sb8fy 1pXnuDYAj8r; do
  python3 generate_allocentric_data.py \
    --scene_dir ../dataset/Matterport/$scene \
    --scene_name $scene \
    --num_samples 100 \
    --output_dir ./data_batch
done
```

### Merge and Split

```bash
python3 merge_allocentric_datasets.py \
  --input_dir ./data_batch \
  --output ./final_dataset.json \
  --split
```

Creates: train/val/test JSON files (80%/10%/10% split)

---

## 🔧 Python API Usage

```python
from src.data_generator import SyntheticDataGenerator

# Initialize
generator = SyntheticDataGenerator(
    scene_name='1LXtFkjw3qL',
    region_csv='path/to/region.csv',
    object_csv='path/to/object.csv',
    step_size=1.0,
    seed=42
)

# Generate
samples = generator.generate_dataset(num_samples=100)

# Save
generator.save_dataset(samples, 'output.json')
```

---

## 📈 Expected Statistics

```
Total samples: 100
Samples with vertical movement: 25%

Action distribution:
  north/south/east/west: ~85%
  up/down: ~10%
  stop: ~5%
```

---

## 🎓 PyTorch Training Example

```python
import json
import torch
from torch.utils.data import Dataset

class NavDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.samples = data['samples']
        self.action_vocab = {
            'north': 0, 'south': 1, 'east': 2, 'west': 3,
            'up': 4, 'down': 5, 'stop': 6
        }
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        action_indices = [self.action_vocab[a] for a in sample['action_sequence']]
        return {
            'instruction': sample['instruction'],
            'actions': torch.LongTensor(action_indices)
        }
```

---

## 📦 Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `networkx>=2.6.0`
- `python>=3.8`

---

## 🔖 Version 2.0.0

Allocentric implementation (October 2025)
