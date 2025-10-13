"""
VLA-3D Synthetic Navigation Data Generator
Allocentric (Absolute Direction) Action Space

This package generates synthetic navigation training data using compass directions.
"""

from .data_types import NavigationSample
from .scene_parser import SceneParser
from .graph_builder import NavigableGraph
from .pathfinder import PathFinder
from .action_converter import ActionConverter
from .instruction_generator import InstructionGenerator
from .data_generator import SyntheticDataGenerator

__version__ = "2.0.0"
__all__ = [
    'NavigationSample',
    'SceneParser',
    'NavigableGraph',
    'PathFinder',
    'ActionConverter',
    'InstructionGenerator',
    'SyntheticDataGenerator'
]
