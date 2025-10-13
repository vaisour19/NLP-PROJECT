"""
Pathfinder - finds optimal paths using A* search
"""
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional


class PathFinder:
    """Step 3: Find paths between nodes using A* search
    
    Uses NetworkX A* implementation with Euclidean distance heuristic.
    """
    
    def __init__(self, graph):
        """
        Args:
            graph: NavigableGraph instance
        """
        self.graph = graph
    
    def find_path(self, start_node: str, goal_node: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using A*
        
        Args:
            start_node: Starting node ID
            goal_node: Goal node ID
            
        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        if not nx.has_path(self.graph.graph, start_node, goal_node):
            return None
        
        try:
            path = nx.astar_path(
                self.graph.graph,
                start_node,
                goal_node,
                heuristic=lambda n1, n2: self._euclidean_heuristic(n1, n2),
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            return None
    
    def _euclidean_heuristic(self, node1: str, node2: str) -> float:
        """Euclidean distance heuristic for A*"""
        coords1 = self.graph.node_info[node1]['coordinates']
        coords2 = self.graph.node_info[node2]['coordinates']
        return np.linalg.norm(coords2 - coords1)
    
    def path_to_coordinates(self, path: List[str]) -> List[Tuple[float, float, float]]:
        """Convert node path to 3D coordinates"""
        coordinates = []
        for node in path:
            coords = self.graph.node_info[node]['coordinates']
            coordinates.append(tuple(coords))
        return coordinates
    
    def calculate_path_distance(self, coordinates: List[Tuple[float, float, float]]) -> float:
        """Calculate total path distance in meters"""
        total_distance = 0.0
        for i in range(len(coordinates) - 1):
            dist = np.linalg.norm(
                np.array(coordinates[i+1]) - np.array(coordinates[i])
            )
            total_distance += dist
        return total_distance
