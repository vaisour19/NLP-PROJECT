"""
Instruction generator - creates natural language descriptions
"""
import random
from typing import List, Dict
from .scene_parser import SceneParser
from .graph_builder import NavigableGraph


class InstructionGenerator:
    """Step 5: Generate natural language instructions
    
    Creates diverse navigation instructions using templates and synonyms.
    """
    
    def __init__(self, scene_parser: SceneParser, graph: NavigableGraph):
        self.parser = scene_parser
        self.graph = graph
        
        # Instruction templates by category
        self.templates = {
            'simple': [
                "Go from the {start} to the {goal}.",
                "Navigate from the {start} to the {goal}.",
                "Walk from the {start} to the {goal}.",
                "Move from the {start} to the {goal}.",
                "Head from the {start} to the {goal}.",
                "Travel from the {start} to the {goal}."
            ],
            'region_to_region': [
                "Starting in the {start}, navigate to the {goal}.",
                "From the {start}, go to the {goal}.",
                "Exit the {start} and enter the {goal}.",
                "Leave the {start} and proceed to the {goal}.",
                "Walk from the {start} into the {goal}."
            ],
            'object_to_object': [
                "Go from the {start} to the {goal}.",
                "Move from the {start} to the {goal}.",
                "Walk from the {start} to the {goal}.",
                "Navigate from the {start} to the {goal}.",
                "Head to the {goal} from the {start}."
            ],
            'with_landmarks': [
                "Go to the {goal}, passing by the {landmark}.",
                "Walk to the {goal}, going past the {landmark}.",
                "Navigate to the {goal} via the {landmark}.",
                "Head to the {goal}, passing the {landmark} along the way.",
                "Move to the {goal}, you'll pass the {landmark}."
            ],
            'vertical': [
                "Go upstairs from the {start} to the {goal}.",
                "Go downstairs from the {start} to the {goal}.",
                "Navigate to the upper floor from the {start} to the {goal}.",
                "Head to the lower level from the {start} to the {goal}.",
                "Take the stairs from the {start} to reach the {goal}."
            ]
        }
    
    def generate_instruction(self, 
                            start_node: str, 
                            goal_node: str,
                            path_nodes: List[str],
                            has_vertical: bool = False) -> str:
        """Generate natural language instruction for navigation task
        
        Args:
            start_node: Starting node ID
            goal_node: Goal node ID
            path_nodes: Full path of node IDs
            has_vertical: Whether path includes vertical movement
            
        Returns:
            Natural language instruction string
        """
        # Get node information
        start_info = self.graph.node_info[start_node]
        goal_info = self.graph.node_info[goal_node]
        
        start_label = self._format_label(start_info['label'])
        goal_label = self._format_label(goal_info['label'])
        
        # Select template category
        if has_vertical and len(path_nodes) > 3:
            category = 'vertical'
        elif len(path_nodes) > 3 and self._has_landmark(path_nodes[1:-1]):
            category = 'with_landmarks'
            landmark_node = random.choice(path_nodes[1:-1])
            landmark_info = self.graph.node_info[landmark_node]
            landmark_label = self._format_label(landmark_info['label'])
        elif start_info['type'] == 'region' and goal_info['type'] == 'region':
            category = 'region_to_region'
        elif start_info['type'] == 'object' or goal_info['type'] == 'object':
            category = 'object_to_object'
        else:
            category = 'simple'
        
        # Select random template
        template = random.choice(self.templates[category])
        
        # Fill in template
        if category == 'with_landmarks':
            instruction = template.format(
                start=start_label,
                goal=goal_label,
                landmark=landmark_label
            )
        else:
            instruction = template.format(
                start=start_label,
                goal=goal_label
            )
        
        return instruction
    
    def _format_label(self, label: str) -> str:
        """Format label for natural language (add article if needed)"""
        # Check if label needs article
        vowels = ['a', 'e', 'i', 'o', 'u']
        
        # Room types don't need article
        room_types = ['kitchen', 'bathroom', 'bedroom', 'hallway', 'office',
                      'living room', 'dining room', 'closet', 'garage']
        
        if label.lower() in room_types:
            return label
        
        # Add appropriate article
        if label[0].lower() in vowels:
            return f"an {label}"
        else:
            return f"a {label}"
    
    def _has_landmark(self, path_nodes: List[str]) -> bool:
        """Check if path contains landmark objects"""
        for node in path_nodes:
            if node.startswith('object_'):
                return True
        return False
    
    def generate_diverse_instructions(self,
                                     start_node: str,
                                     goal_node: str, 
                                     path_nodes: List[str],
                                     has_vertical: bool = False,
                                     num_variants: int = 3) -> List[str]:
        """Generate multiple instruction variants for the same path
        
        Useful for data augmentation.
        
        Args:
            start_node: Starting node ID
            goal_node: Goal node ID
            path_nodes: Full path of node IDs
            has_vertical: Whether path includes vertical movement
            num_variants: Number of variants to generate
            
        Returns:
            List of instruction strings
        """
        instructions = []
        for _ in range(num_variants):
            instruction = self.generate_instruction(
                start_node, goal_node, path_nodes, has_vertical
            )
            if instruction not in instructions:
                instructions.append(instruction)
        
        return instructions
