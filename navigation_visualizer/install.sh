#!/bin/bash
# Quick start script for navigation visualizer

echo "================================"
echo "Navigation Visualizer Quick Start"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "navigation_visualizer.py" ]; then
    echo "Error: Please run this from the navigation_visualizer directory"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "Now you can run the visualizer with:"
echo ""
echo "  python navigation_visualizer.py \\"
echo "      --scene_path ../dataset/Matterport/1LXtFkjw3qL \\"
echo "      --navigation_data ../navigation_generator/allocentric_data/1LXtFkjw3qL_allocentric_data.json"
echo ""
echo "Or use the example script:"
echo "  bash run_example.sh"
echo ""
