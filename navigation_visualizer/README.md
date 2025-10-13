# 🚀 Navigation Visualizer (PyVista)

A powerful, interactive 3D visualizer for synthetic navigation datasets with full transparency control, color-coded action arrows, and comprehensive scene visualization.

![Status](https://img.shields.io/badge/status-ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Built with PyVista for superior transparency and interactive controls!**

## Features

### Visualization Components
- **Point Cloud**: Scene point cloud with adjustable transparency
- **Region Bounding Boxes**: Wireframe boxes showing room boundaries
- **Object Bounding Boxes**: Colored boxes for navigable objects (furniture, etc.)
- **Trajectory Path**: Smooth tube showing the navigation path
- **Waypoint Markers**: Cyan spheres at path waypoints
- **Action Arrows**: Color-coded arrows showing each navigation action
  - 🔴 **Red**: North
  - 🔵 **Blue**: South
  - 🟢 **Green**: East
  - 🟡 **Yellow**: West
  - 🟣 **Magenta**: Up
  - 🔵 **Cyan**: Down
- **Start/Goal Markers**: Large green (start) and red (goal) spheres with labels

### Interactive Controls
- **Sample Navigation**: Browse through all navigation samples
- **Visibility Toggles**: Show/hide individual components
- **Transparency Sliders**: Adjust opacity of all scene elements
- **Sample Information**: View instruction, statistics, and metadata

## Installation

```bash
cd navigation_visualizer
pip install -r requirements.txt
```

**Requirements:**
- PyVista >= 0.43.0
- PyVistaQt >= 0.11.0
- PyQt5 >= 5.15.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0

## Usage

### Basic Usage

```bash
python navigation_visualizer.py \
    --scene_path ../dataset/Matterport/1LXtFkjw3qL \
    --navigation_data ../navigation_generator/allocentric_data/1LXtFkjw3qL_allocentric_data.json
```

### With Custom Step Size

```bash
python navigation_visualizer.py \
    --scene_path ../dataset/Matterport/1LXtFkjw3qL \
    --navigation_data ../navigation_generator/allocentric_data/1LXtFkjw3qL_allocentric_data.json \
    --step_size 1.5
```

## Controls

### Sample Navigation
- **Spinner**: Directly select sample number
- **Previous/Next Buttons**: Navigate through samples sequentially
- **Instruction Display**: Shows natural language instruction
- **Statistics Panel**: Displays sample metadata

### Visibility Controls
Toggle visibility for:
- Point Cloud
- Region Boxes
- Object Boxes
- Trajectory
- Waypoints
- Action Arrows
- Start Marker
- Goal Marker

### Transparency Sliders
Adjust opacity (0-100%) for:
- Point Cloud
- Region Boxes
- Object Boxes
- Trajectory
- Action Arrows

### Camera Controls
- **Left Click + Drag**: Rotate camera
- **Right Click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Middle Click**: Reset camera view

## Action Color Coding

Actions are visualized as colored arrows:

| Action | Color | Direction |
|--------|-------|-----------|
| North | Red | +Y axis |
| South | Blue | -Y axis |
| East | Green | +X axis |
| West | Yellow | -X axis |
| Up | Magenta | +Z axis |
| Down | Cyan | -Z axis |

## Example Workflow

1. **Generate navigation data**:
   ```bash
   cd ../navigation_generator
   python generate_allocentric_data.py \
       --scene_dir ../dataset/Matterport/1LXtFkjw3qL \
       --scene_name 1LXtFkjw3qL \
       --num_samples 100 \
       --output_dir ./allocentric_data
   ```

2. **Visualize the data**:
   ```bash
   cd ../navigation_visualizer
   python navigation_visualizer.py \
       --scene_path ../dataset/Matterport/1LXtFkjw3qL \
       --navigation_data ../navigation_generator/allocentric_data/1LXtFkjw3qL_allocentric_data.json
   ```

3. **Explore the visualization**:
   - Use Previous/Next buttons to browse samples
   - Toggle visibility to focus on specific elements
   - Adjust transparency to see overlapping structures
   - Rotate and zoom to examine trajectories from different angles

## File Structure

```
navigation_visualizer/
├── navigation_visualizer.py  # Main visualizer script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Advantages Over Open3D

1. **Better Transparency**: PyVista handles translucent surfaces much better
2. **Interactive UI**: Integrated Qt controls with sliders and checkboxes
3. **Color-coded Actions**: Easy to distinguish different action types
4. **Smooth Trajectories**: Spline-based path rendering
5. **Better Performance**: Efficient rendering of complex scenes
6. **More Intuitive**: Easier camera controls and interaction

## Troubleshooting

### Import Errors
If you get Qt-related errors:
```bash
pip install --upgrade PyQt5 pyvistaqt
```

### Rendering Issues
If visualization is slow or buggy:
- Reduce point cloud density
- Hide regions/objects while browsing samples
- Lower transparency values

### Point Cloud Not Loading
- Ensure `.ply` file exists in scene directory
- Check file is named `{scene_name}_pc_result.ply`
- Verify PLY file is valid with: `pv.read('path/to/file.ply')`

## Tips

1. **Focus on Trajectory**: Hide regions/objects to clearly see the navigation path
2. **Examine Spatial Relationships**: Set regions to low opacity (10-20%) to see how paths navigate through rooms
3. **Verify Actions**: Use arrow colors to confirm action sequences match instructions
4. **Compare Samples**: Quickly browse samples to find interesting navigation patterns
5. **Performance**: Hide point cloud for faster interaction when examining many samples

## Future Enhancements

Potential additions:
- [ ] Export visualizations as images/videos
- [ ] Side-by-side comparison of multiple samples
- [ ] Filter samples by criteria (length, vertical movement, etc.)
- [ ] Animate trajectory playback
- [ ] Show egocentric view from agent perspective
- [ ] Display action sequence as timeline
- [ ] Heat map of visited regions

## Credits

Built for the VLA-3D synthetic navigation dataset generator.
