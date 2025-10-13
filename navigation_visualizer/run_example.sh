#!/bin/bash
# Example: Run visualizer with the 1LXtFkjw3qL scene

SCENE_PATH="../dataset/Matterport/1LXtFkjw3qL"
NAV_DATA="../navigation_generator/allocentric_data/1LXtFkjw3qL_allocentric_data.json"

echo "Running Navigation Visualizer..."
echo "Scene: $SCENE_PATH"
echo "Data: $NAV_DATA"
echo ""

# Check if files exist
if [ ! -d "$SCENE_PATH" ]; then
    echo "Error: Scene path not found: $SCENE_PATH"
    echo "Please update SCENE_PATH in this script"
    exit 1
fi

if [ ! -f "$NAV_DATA" ]; then
    echo "Error: Navigation data not found: $NAV_DATA"
    echo ""
    echo "Generate it first with:"
    echo "  cd ../navigation_generator"
    echo "  python generate_allocentric_data.py \\"
    echo "      --scene_dir ../dataset/Matterport/1LXtFkjw3qL \\"
    echo "      --scene_name 1LXtFkjw3qL \\"
    echo "      --num_samples 100 \\"
    echo "      --output_dir ./allocentric_data"
    exit 1
fi

# Run visualizer
python navigation_visualizer.py \
    --scene_path "$SCENE_PATH" \
    --navigation_data "$NAV_DATA" \
    --step_size 1.0
