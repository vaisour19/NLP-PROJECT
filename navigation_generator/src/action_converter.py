"""
Action converter - converts paths to allocentric (absolute) action sequences
"""
import numpy as np
from typing import List, Tuple, Dict


class ActionConverter:
    """Step 4: Convert path coordinates to ALLOCENTRIC action sequences
    
    Action Space: ALLOCENTRIC (Absolute Compass Directions)
    -------------------------------------------------------
    Actions use absolute world directions, independent of agent orientation:
    
    - 'north': Move in +Y direction (absolute north)
    - 'south': Move in -Y direction (absolute south)
    - 'east': Move in +X direction (absolute east)
    - 'west': Move in -X direction (absolute west)
    - 'up': Move in +Z direction (vertically upward)
    - 'down': Move in -Z direction (vertically downward)
    - 'stop': Reach goal location
    
    Coordinate System:
        X-axis: East (+X) / West (-X)
        Y-axis: North (+Y) / South (-Y)
        Z-axis: Up (+Z) / Down (-Z)
    
    Example Path:
        Start: (0, 0, 0)
        Step 1: (0, 5, 0)  → Actions: ['north', 'north', 'north', 'north', 'north']
        Step 2: (3, 5, 0)  → Actions: ['east', 'east', 'east']
        Step 3: (3, 5, 2)  → Actions: ['up', 'up']
        Final: (3, 5, 2)   → Actions: ['stop']
        
    Complete sequence: ['north', 'north', 'north', 'north', 'north', 
                        'east', 'east', 'east', 'up', 'up', 'stop']
    
    Benefits of Allocentric Actions:
    - No heading state required
    - Simpler action space
    - Direct correspondence with world coordinates
    - Easier to visualize and debug
    - Natural for gridworld and map-based representations
    """
    
    def __init__(self, 
                 step_size: float = 1.0,
                 vertical_threshold: float = 0.5):
        """
        Args:
            step_size: Distance per action (meters). Each action moves this distance.
            vertical_threshold: Minimum Z-diff to trigger vertical movement (meters)
        """
        self.step_size = step_size
        self.vertical_threshold = vertical_threshold
    
    def path_to_actions(self, coordinates: List[Tuple[float, float, float]]) -> List[str]:
        """Convert path coordinates to allocentric action sequence
        
        Algorithm:
        1. For each waypoint transition, calculate movement vector (dx, dy, dz)
        2. Determine dominant direction (N/S/E/W/U/D)
        3. Emit multiple actions to cover the distance (distance / step_size)
        4. Prioritize vertical movement when significant
        5. Add 'stop' at the end
        
        Args:
            coordinates: List of (x, y, z) waypoint tuples
            
        Returns:
            List of action strings ['north', 'east', 'up', 'stop']
        """
        if len(coordinates) < 2:
            return ['stop']
        
        actions = []
        
        for i in range(len(coordinates) - 1):
            current_pos = np.array(coordinates[i])
            next_pos = np.array(coordinates[i + 1])
            
            # Calculate movement vector
            movement = next_pos - current_pos
            dx, dy, dz = movement[0], movement[1], movement[2]
            
            # Calculate distances
            horizontal_dist = np.sqrt(dx**2 + dy**2)
            vertical_dist = abs(dz)
            
            # Check for significant vertical movement
            if vertical_dist > self.vertical_threshold and vertical_dist > horizontal_dist:
                # Vertical movement dominates
                num_steps = max(1, int(round(vertical_dist / self.step_size)))
                
                if dz > 0:
                    actions.extend(['up'] * num_steps)
                else:
                    actions.extend(['down'] * num_steps)
            
            elif horizontal_dist > 0.1:  # Significant horizontal movement
                # Break down into cardinal directions
                
                # North/South component (Y-axis)
                if abs(dy) > 0.1:
                    num_steps = max(1, int(round(abs(dy) / self.step_size)))
                    if dy > 0:
                        actions.extend(['north'] * num_steps)
                    else:
                        actions.extend(['south'] * num_steps)
                
                # East/West component (X-axis)
                if abs(dx) > 0.1:
                    num_steps = max(1, int(round(abs(dx) / self.step_size)))
                    if dx > 0:
                        actions.extend(['east'] * num_steps)
                    else:
                        actions.extend(['west'] * num_steps)
        
        # Add stop action
        actions.append('stop')
        return actions
    
    def analyze_vertical_movement(self, coordinates: List[Tuple[float, float, float]]) -> Dict:
        """Analyze vertical movement characteristics in path
        
        Returns:
            Dictionary with vertical movement statistics
        """
        z_values = [coord[2] for coord in coordinates]
        
        total_vertical = 0.0
        for i in range(len(z_values) - 1):
            total_vertical += abs(z_values[i+1] - z_values[i])
        
        return {
            'has_vertical': total_vertical > self.vertical_threshold,
            'total_vertical_distance': total_vertical,
            'z_range': max(z_values) - min(z_values),
            'min_z': min(z_values),
            'max_z': max(z_values)
        }
    
    def get_action_statistics(self, actions: List[str]) -> Dict:
        """Calculate statistics about action sequence
        
        Returns:
            Dictionary with action counts and percentages
        """
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total = len(actions)
        action_percentages = {
            action: (count / total * 100) for action, count in action_counts.items()
        }
        
        return {
            'counts': action_counts,
            'percentages': action_percentages,
            'total_actions': total
        }
