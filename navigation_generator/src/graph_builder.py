"""
Graph builder - creates navigable graph from scene data
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Set
from .scene_parser import SceneParser


class NavigableGraph:
    """Step 2: Create a navigable graph with nodes and edges
    
    Creates a NetworkX graph where:
    - Nodes: Region centers + landmark objects
    - Edges: Connections between spatially adjacent nodes
    - Weights: 3D Euclidean distances
    """
    
    def __init__(self, scene_parser: SceneParser, 
                 region_distance_threshold: float = 2.0,
                 object_distance_threshold: float = 5.0):
        self.parser = scene_parser
        self.region_distance_threshold = region_distance_threshold
        self.object_distance_threshold = object_distance_threshold
        
        self.graph = nx.Graph()
        self.node_info = {}  # node_id -> info dict
        
        self._build_graph()
    
    def _build_graph(self):
        """Build the navigation graph"""
        # Add region nodes (use FLOOR-LEVEL coordinates, not bbox center)
        for region_id, region_data in self.parser.regions.items():
            node_id = f"region_{region_id}"
            self.graph.add_node(node_id)
            
            # Calculate floor level (bottom of bbox, not center)
            center = region_data['center']
            size = region_data['size']
            floor_z = center[2] - (size[2] / 2.0)  # Bottom of bbox (z is index 2)
            floor_coords = np.array([center[0], center[1], floor_z])
            
            self.node_info[node_id] = {
                'type': 'region',
                'id': node_id,
                'label': region_data['label'],
                'coordinates': floor_coords,  # Use floor level!
                'region_id': region_id,
                'is_stairs': 'stair' in region_data['label'].lower()
            }
        
        # Add landmark object nodes (also use floor level)
        landmark_objects = self._get_landmark_objects()
        for object_id in landmark_objects:
            obj_data = self.parser.objects[object_id]
            node_id = f"object_{object_id}"
            self.graph.add_node(node_id)
            
            # Calculate floor level for objects too
            center = obj_data['center']
            size = obj_data['size']
            floor_z = center[2] - (size[2] / 2.0)  # Bottom of bbox
            floor_coords = np.array([center[0], center[1], floor_z])
            
            self.node_info[node_id] = {
                'type': 'object',
                'id': node_id,
                'label': obj_data['label'],
                'coordinates': floor_coords,  # Use floor level!
                'region_id': obj_data['region_id']
            }
        
        # Connect nodes
        self._connect_regions()
        self._connect_objects_to_regions()
        
        print(f"✓ Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _get_landmark_objects(self) -> List[str]:
        """Get list of landmark object IDs
        
        Landmark objects are furniture and fixtures that serve as
        useful navigation waypoints.
        """
        landmark_categories = [
            'bed', 'sofa', 'chair', 'table', 'desk', 'cabinet', 'shelving',
            'door', 'stairs', 'counter', 'sink', 'toilet', 'bathtub', 'shower',
            'refrigerator', 'stove', 'oven', 'microwave', 'dishwasher',
            'washing machine', 'tv', 'lamp', 'plant', 'picture', 'mirror'
        ]
        
        landmarks = []
        for object_id, obj_data in self.parser.objects.items():
            if obj_data['label'] in landmark_categories:
                landmarks.append(object_id)
        
        return landmarks
    
    def _connect_regions(self):
        """Connect spatially adjacent regions, respecting floor constraints"""
        region_nodes = [n for n in self.graph.nodes() if n.startswith('region_')]
        
        # Increase threshold for better connectivity
        HORIZONTAL_THRESHOLD = 8.0  # More lenient for horizontal distance
        SAME_FLOOR_Z_THRESHOLD = 2.0  # Within 2m vertically = same floor
        STAIR_CONNECTION_THRESHOLD = 10.0  # Stairs can connect further
        
        for i, node1 in enumerate(region_nodes):
            for node2 in region_nodes[i+1:]:
                coords1 = self.node_info[node1]['coordinates']
                coords2 = self.node_info[node2]['coordinates']
                
                # Calculate distances
                horizontal_dist = np.sqrt((coords2[0] - coords1[0])**2 + 
                                         (coords2[1] - coords1[1])**2)
                vertical_dist = abs(coords2[2] - coords1[2])
                total_dist = np.linalg.norm(coords2 - coords1)
                
                # Check if either node is stairs
                is_stairs1 = self.node_info[node1].get('is_stairs', False)
                is_stairs2 = self.node_info[node2].get('is_stairs', False)
                
                # Connection rules:
                # 1. Same floor (small vertical difference) + close horizontally
                if vertical_dist < SAME_FLOOR_Z_THRESHOLD and horizontal_dist < HORIZONTAL_THRESHOLD:
                    self.graph.add_edge(node1, node2, weight=total_dist)
                
                # 2. One or both are stairs + reasonably close
                elif (is_stairs1 or is_stairs2) and total_dist < STAIR_CONNECTION_THRESHOLD:
                    # Add penalty for vertical movement to prefer horizontal paths
                    weight = horizontal_dist + (vertical_dist * 2.0)  # 2x penalty for vertical
                    self.graph.add_edge(node1, node2, weight=weight)
    
    def _connect_objects_to_regions(self):
        """Connect objects to their parent regions and nearby objects"""
        object_nodes = [n for n in self.graph.nodes() if n.startswith('object_')]
        region_nodes = [n for n in self.graph.nodes() if n.startswith('region_')]
        
        for obj_node in object_nodes:
            obj_coords = self.node_info[obj_node]['coordinates']
            obj_region_id = self.node_info[obj_node]['region_id']
            
            # Connect to parent region
            parent_region_node = f"region_{obj_region_id}"
            if parent_region_node in self.graph:
                region_coords = self.node_info[parent_region_node]['coordinates']
                distance = np.linalg.norm(obj_coords - region_coords)
                self.graph.add_edge(obj_node, parent_region_node, weight=distance)
            
            # Connect to nearby objects in same region
            for other_obj_node in object_nodes:
                if obj_node == other_obj_node:
                    continue
                
                other_region_id = self.node_info[other_obj_node]['region_id']
                if other_region_id != obj_region_id:
                    continue
                
                other_coords = self.node_info[other_obj_node]['coordinates']
                distance = np.linalg.norm(other_coords - obj_coords)
                
                if distance < self.object_distance_threshold:
                    self.graph.add_edge(obj_node, other_obj_node, weight=distance)
    
    def get_random_nodes(self, num_pairs: int = 1) -> List[tuple]:
        """Get random pairs of connected nodes"""
        pairs = []
        nodes = list(self.graph.nodes())
        
        for _ in range(num_pairs * 10):  # Try multiple times
            start = np.random.choice(nodes)
            goal = np.random.choice(nodes)
            
            if start != goal and nx.has_path(self.graph, start, goal):
                pairs.append((start, goal))
                if len(pairs) >= num_pairs:
                    break
        
        return pairs
