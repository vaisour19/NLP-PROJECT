"""
Main synthetic data generator - orchestrates the full pipeline
"""
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from .data_types import NavigationSample
from .scene_parser import SceneParser
from .graph_builder import NavigableGraph
from .pathfinder import PathFinder
from .action_converter import ActionConverter
from .instruction_generator import InstructionGenerator


class SyntheticDataGenerator:
    """Main pipeline orchestrator for synthetic navigation data generation
    
    Implements the complete 5-step pipeline:
    1. Parse Scene Data
    2. Create Navigable Graph
    3. Find Paths
    4. Convert to Actions (ALLOCENTRIC)
    5. Generate Instructions
    """
    
    def __init__(self,
                 scene_name: str,
                 region_csv: str,
                 object_csv: str,
                 step_size: float = 1.0,
                 seed: int = 42):
        """
        Args:
            scene_name: Scene ID
            region_csv: Path to region CSV file
            object_csv: Path to object CSV file
            step_size: Distance per action in meters (for allocentric actions)
            seed: Random seed for reproducibility
        """
        self.scene_name = scene_name
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"Initializing Synthetic Data Generator (ALLOCENTRIC)")
        print(f"Scene: {scene_name}")
        print(f"Action Space: Allocentric (North/South/East/West/Up/Down)")
        print(f"Step Size: {step_size}m per action")
        print(f"{'='*80}\n")
        
        # Step 1: Parse scene data
        print("Step 1: Parsing scene data...")
        self.scene_parser = SceneParser(region_csv, object_csv, scene_name)
        
        # Step 2: Build navigable graph
        print("\nStep 2: Building navigable graph...")
        self.graph = NavigableGraph(self.scene_parser)
        
        # Step 3: Initialize pathfinder
        print("\nStep 3: Initializing A* pathfinder...")
        self.pathfinder = PathFinder(self.graph)
        
        # Step 4: Initialize action converter
        print("\nStep 4: Initializing action converter (ALLOCENTRIC)...")
        self.action_converter = ActionConverter(step_size=step_size)
        
        # Step 5: Initialize instruction generator
        print("\nStep 5: Initializing instruction generator...")
        self.instruction_generator = InstructionGenerator(
            self.scene_parser, self.graph
        )
        
        print(f"\n{'='*80}")
        print("✓ Initialization complete!")
        print(f"{'='*80}\n")
    
    def generate_sample(self, sample_id: int) -> NavigationSample:
        """Generate a single navigation sample
        
        Args:
            sample_id: Unique sample identifier
            
        Returns:
            NavigationSample instance
        """
        # Get random start and goal nodes
        pairs = self.graph.get_random_nodes(num_pairs=1)
        if not pairs:
            return None
        
        start_node, goal_node = pairs[0]
        
        # Find path
        path_nodes = self.pathfinder.find_path(start_node, goal_node)
        if not path_nodes:
            return None
        
        # Convert to coordinates
        path_coords = self.pathfinder.path_to_coordinates(path_nodes)
        
        # Convert to actions (ALLOCENTRIC)
        actions = self.action_converter.path_to_actions(path_coords)
        
        # Analyze vertical movement
        vertical_info = self.action_converter.analyze_vertical_movement(path_coords)
        
        # Generate instruction
        instruction = self.instruction_generator.generate_instruction(
            start_node, goal_node, path_nodes, 
            has_vertical=vertical_info['has_vertical']
        )
        
        # Calculate path metrics
        path_length = self.pathfinder.calculate_path_distance(path_coords)
        
        # Get region IDs
        start_info = self.graph.node_info[start_node]
        goal_info = self.graph.node_info[goal_node]
        
        # Extract landmark objects from path
        landmarks = []
        for node in path_nodes[1:-1]:  # Exclude start and goal
            if node.startswith('object_'):
                obj_label = self.graph.node_info[node]['label']
                landmarks.append(obj_label)
        
        # Create sample
        sample = NavigationSample(
            task_id=f"{self.scene_name}_synthetic_{sample_id:04d}",
            scene_name=self.scene_name,
            instruction=instruction,
            action_sequence=actions,
            path_coordinates=path_coords,
            start_node=start_node,
            goal_node=goal_node,
            start_region=start_info['region_id'],
            goal_region=goal_info['region_id'],
            path_length=path_length,
            num_actions=len(actions),
            has_vertical_movement=vertical_info['has_vertical'],
            vertical_distance=vertical_info['total_vertical_distance'],
            landmarks=landmarks
        )
        
        return sample
    
    def generate_dataset(self, num_samples: int) -> List[NavigationSample]:
        """Generate a complete dataset
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of NavigationSample instances
        """
        print(f"Generating {num_samples} navigation samples...")
        print(f"{'='*80}\n")
        
        samples = []
        attempts = 0
        max_attempts = num_samples * 3
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            sample = self.generate_sample(len(samples))
            
            if sample:
                samples.append(sample)
                
                if len(samples) % 10 == 0:
                    print(f"  Generated {len(samples)}/{num_samples} samples...")
        
        print(f"\n✓ Successfully generated {len(samples)} samples")
        print(f"  (Attempts: {attempts})")
        
        return samples
    
    def save_dataset(self, samples: List[NavigationSample], output_path: str):
        """Save dataset to JSON file
        
        Args:
            samples: List of NavigationSample instances
            output_path: Path to output JSON file
        """
        # Convert samples to dictionaries
        samples_dict = [sample.to_dict() for sample in samples]
        
        # Create output data
        output_data = {
            'scene_name': self.scene_name,
            'num_samples': len(samples),
            'action_space': 'allocentric',
            'actions': ['north', 'south', 'east', 'west', 'up', 'down', 'stop'],
            'coordinate_system': {
                'x_axis': 'East (positive) / West (negative)',
                'y_axis': 'North (positive) / South (negative)',
                'z_axis': 'Up (positive) / Down (negative)'
            },
            'samples': samples_dict
        }
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Dataset saved to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")
    
    def print_statistics(self, samples: List[NavigationSample]):
        """Print dataset statistics
        
        Args:
            samples: List of NavigationSample instances
        """
        if not samples:
            print("No samples to analyze")
            return
        
        print(f"\n{'='*80}")
        print("Dataset Statistics")
        print(f"{'='*80}\n")
        
        print(f"Total samples: {len(samples)}")
        
        # Count samples with vertical movement
        vertical_samples = [s for s in samples if s.has_vertical_movement]
        print(f"Samples with vertical movement: {len(vertical_samples)} "
              f"({len(vertical_samples)/len(samples)*100:.1f}%)")
        
        # Average statistics
        avg_actions = np.mean([s.num_actions for s in samples])
        avg_path_length = np.mean([s.path_length for s in samples])
        
        print(f"\nAverage statistics:")
        print(f"  Actions per sample: {avg_actions:.1f}")
        print(f"  Path length: {avg_path_length:.2f}m")
        
        if vertical_samples:
            avg_vertical = np.mean([s.vertical_distance for s in vertical_samples])
            print(f"  Vertical distance (when present): {avg_vertical:.2f}m")
        
        # Action distribution
        all_actions = []
        for sample in samples:
            all_actions.extend(sample.action_sequence)
        
        action_counts = defaultdict(int)
        for action in all_actions:
            action_counts[action] += 1
        
        total_actions = len(all_actions)
        print(f"\nAction distribution:")
        for action in ['north', 'south', 'east', 'west', 'up', 'down', 'stop']:
            count = action_counts.get(action, 0)
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"  {action:10s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n{'='*80}\n")
    
    def print_examples(self, samples: List[NavigationSample], num_examples: int = 5):
        """Print example samples
        
        Args:
            samples: List of NavigationSample instances
            num_examples: Number of examples to print
        """
        print(f"\n{'='*80}")
        print(f"Example Samples (showing first {num_examples})")
        print(f"{'='*80}\n")
        
        for i, sample in enumerate(samples[:num_examples]):
            print(f"--- Example {i+1} ---")
            print(f"Task ID: {sample.task_id}")
            print(f"Instruction: {sample.instruction}")
            print(f"Action Sequence: {' -> '.join(sample.action_sequence[:10])}")
            if len(sample.action_sequence) > 10:
                print(f"                 ... ({len(sample.action_sequence)} actions total)")
            print(f"Path Length: {sample.path_length:.2f}m")
            print(f"Num Actions: {sample.num_actions}")
            print(f"Vertical Movement: {'Yes' if sample.has_vertical_movement else 'No'}")
            if sample.has_vertical_movement:
                print(f"Vertical Distance: {sample.vertical_distance:.2f}m")
            if sample.landmarks:
                print(f"Landmarks: {', '.join(sample.landmarks)}")
            print()
