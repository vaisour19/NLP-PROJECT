"""
Data structures for navigation samples
"""
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np


@dataclass
class NavigationSample:
    """A single training sample: (instruction, action_sequence) pair
    
    Action Space: ALLOCENTRIC (Absolute Compass Directions)
    -------------------------------------------------------
    Actions use absolute compass directions:
    - 'north': Move north (positive Y direction)
    - 'south': Move south (negative Y direction)  
    - 'east': Move east (positive X direction)
    - 'west': Move west (negative X direction)
    - 'up': Move vertically upward
    - 'down': Move vertically downward
    - 'stop': Reach goal location
    
    These actions are independent of agent heading and always refer
    to absolute world directions.
    """
    task_id: str
    scene_name: str
    instruction: str
    action_sequence: List[str]  # ['north', 'east', 'up', 'stop']
    path_coordinates: List[Tuple[float, float, float]]
    start_node: str
    goal_node: str
    start_region: str
    goal_region: str
    path_length: float
    num_actions: int
    has_vertical_movement: bool
    vertical_distance: float
    landmarks: List[str]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization
        
        Ensures all numpy types are converted to Python native types
        """
        data = asdict(self)
        
        # Convert numpy types to Python native types
        def convert_value(val):
            # Check for numpy scalar types using numpy's generic base classes
            if isinstance(val, (np.integer, int)):
                return int(val) if not isinstance(val, bool) else bool(val)
            elif isinstance(val, np.floating):
                return float(val)
            elif isinstance(val, np.bool_):
                return bool(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (list, tuple)):
                return [convert_value(item) for item in val]
            elif isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            else:
                return val
        
        return {k: convert_value(v) for k, v in data.items()}
