"""
Path planning algorithms for Eco Maps.
"""

import networkx as nx
from typing import List, Tuple, Dict, Optional
import numpy as np
from eco_maps.osm_loader import OSMRoadNetwork


class PathPlanner:
    """Plan routes using graph algorithms."""
    
    def __init__(self, road_network: OSMRoadNetwork):
        self.network = road_network
        self.graph = road_network.graph
    
    def find_k_shortest_paths(self, origin: Tuple[float, float],
                             destination: Tuple[float, float],
                             k: int = 5) -> List[Dict]:
        """Find K shortest paths (alternative routes)."""
        orig_node = self.network.get_nearest_node(origin[0], origin[1])
        dest_node = self.network.get_nearest_node(destination[0], destination[1])
        
        try:
            paths = list(nx.shortest_simple_paths(
                self.graph,
                orig_node,
                dest_node,
                weight='length'
            ))[:k]
        except nx.NetworkXNoPath:
            return []
        
        routes = [self._path_to_route(path) for path in paths]
        return routes
    
    def find_astar_path(self, origin: Tuple[float, float],
                       destination: Tuple[float, float]) -> Dict:
        """Find path using A* algorithm."""
        orig_node = self.network.get_nearest_node(origin[0], origin[1])
        dest_node = self.network.get_nearest_node(destination[0], destination[1])
        dest_coords = self.network.get_node_coords(dest_node)
        
        def h(u, v):
            u_coords = self.network.get_node_coords(u)
            return self._haversine_distance(u_coords, dest_coords)
        
        try:
            path = nx.astar_path(
                self.graph,
                orig_node,
                dest_node,
                heuristic=h,
                weight='length'
            )
        except nx.NetworkXNoPath:
            return None
        
        return self._path_to_route(path)
    
    def _path_to_route(self, path: List[int]) -> Dict:
        """Convert node path to route dictionary."""
        segments = []
        total_distance = 0
        total_time = 0
        total_elevation_gain = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            edge_data = None
            for key in self.graph[u][v]:
                edge_data = self.graph[u][v][key]
                break
            
            if edge_data:
                u_coords = self.network.get_node_coords(u)
                v_coords = self.network.get_node_coords(v)
                
                length = edge_data.get('length', 0)
                travel_time = edge_data.get('travel_time', 0)
                grade = edge_data.get('grade', 0)
                
                elevation_gain = max(0, length * grade)
                
                segments.append({
                    'start': list(u_coords),
                    'end': list(v_coords),
                    'distance_m': length,
                    'elevation_gain_m': elevation_gain,
                    'travel_time_s': travel_time,
                    'road_type': edge_data.get('highway', 'unknown'),
                    'has_bike_lane': 'cycleway' in str(edge_data.get('highway', '')),
                    'maxspeed': edge_data.get('maxspeed', 50)
                })
                
                total_distance += length
                total_time += travel_time
                total_elevation_gain += elevation_gain
        
        return {
            'route_id': f"route_osm_{hash(tuple(path)) % 10000:04d}",
            'segments': segments,
            'total_distance_m': total_distance,
            'estimated_travel_time_s': total_time,
            'total_elevation_gain_m': total_elevation_gain,
            'node_path': path
        }
    
    @staticmethod
    def _haversine_distance(coord1: Tuple[float, float], 
                           coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two points."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
