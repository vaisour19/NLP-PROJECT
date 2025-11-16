#!/usr/bin/env python3
"""
Convenience script to run floor decomposition pipeline
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import as a package
import floor_detector
import dataset_splitter
import visualizer
import pipeline

if __name__ == '__main__':
    pipeline.main()
