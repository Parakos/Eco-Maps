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
                             k: int = 3,
                             mode: str = 'drive') -> List[Dict]:
        """
        Find K alternative routes using penalty method (MUCH faster than Yen's algorithm).
        
        This finds alternative routes by:
        1. Finding shortest path
        2. Penalizing edges on that path
        3. Finding next shortest path (which will avoid previous routes)
        4. Repeat
        
        This is 10-100x faster than nx.shortest_simple_paths for large graphs.
        """
        print(f"    Finding nearest nodes...")
        orig_node = self.network.get_nearest_node(origin[0], origin[1])
        dest_node = self.network.get_nearest_node(destination[0], destination[1])
        
        orig_coords = self.network.get_node_coords(orig_node)
        dest_coords = self.network.get_node_coords(dest_node)
        
        straight_line_dist = self._haversine_distance(orig_coords, dest_coords)
        
        print(f"    Origin node: {orig_node} at {orig_coords}")
        print(f"    Destination node: {dest_node} at {dest_coords}")
        print(f"    Straight-line distance: {straight_line_dist/1000:.2f} km")
        
        # Check if nodes are the same
        if orig_node == dest_node:
            print(f"    ⚠️  Origin and destination map to same node!")
            print(f"    Try using locations that are further apart (>1 km)")
            return []
        
        # Check if path exists
        if not nx.has_path(self.graph, orig_node, dest_node):
            print(f"    ⚠️  No path exists between nodes")
            return []
        
        routes = []
        edge_penalties = {}  # Track which edges we've penalized
        penalty_factor = 1.5  # How much to penalize used edges
        
        for i in range(k):
            print(f"    Calculating route {i+1}/{k}...")
            
            try:
                # Find shortest path with current penalties
                path = nx.shortest_path(
                    self.graph,
                    orig_node,
                    dest_node,
                    weight=lambda u, v, d: self._get_penalized_weight(
                        u, v, d, edge_penalties, penalty_factor
                    )
                )
                
                # Convert to route
                route = self._path_to_route(path, route_num=i+1, mode=mode)
                routes.append(route)
                
                print(f"      ✓ Route {i+1}: {route['total_distance_m']/1000:.2f} km, "
                      f"{route['estimated_travel_time_s']/60:.1f} min")
                
                # Penalize edges in this path for next iteration
                for j in range(len(path) - 1):
                    edge = (path[j], path[j+1])
                    edge_penalties[edge] = edge_penalties.get(edge, 0) + 1
                
            except nx.NetworkXNoPath:
                print(f"    ⚠️  Could not find {k} distinct paths (found {i})")
                break
        
        return routes
    
    def _get_penalized_weight(self, u, v, edge_data, penalties, penalty_factor):
        """Get edge weight with penalties applied."""
        base_weight = edge_data.get('length', 1)
        
        # Check both directions since graph might be directed
        penalty = penalties.get((u, v), 0) + penalties.get((v, u), 0)
        
        return base_weight * (penalty_factor ** penalty)
    
    def find_shortest_path(self, origin: Tuple[float, float],
                          destination: Tuple[float, float]) -> Optional[Dict]:
        """Find single shortest path using Dijkstra."""
        print(f"    Finding nearest nodes...")
        orig_node = self.network.get_nearest_node(origin[0], origin[1])
        dest_node = self.network.get_nearest_node(destination[0], destination[1])
        
        print(f"    Calculating shortest path...")
        
        try:
            path = nx.shortest_path(
                self.graph,
                orig_node,
                dest_node,
                weight='length'
            )
        except nx.NetworkXNoPath:
            print(f"    ⚠️  No path found")
            return None
        
        route = self._path_to_route(path, route_num=1)
        print(f"      ✓ Route: {route['total_distance_m']/1000:.2f} km, "
              f"{route['estimated_travel_time_s']/60:.1f} min")
        
        return route
    
    def find_path_astar(self, origin: Tuple[float, float],
                       destination: Tuple[float, float]) -> Optional[Dict]:
        """Find path using A* algorithm (faster for long distances)."""
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
    
    def _path_to_route(self, path: List[int], route_num: int = 1, mode: str = 'drive') -> Dict:
        """Convert node path to route dictionary."""
        
        # Debug: check path length
        if len(path) < 2:
            print(f"      ⚠️  Warning: Path has only {len(path)} node(s)")
        
        segments = []
        total_distance = 0
        total_time = 0
        total_elevation_gain = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get edge data (handle MultiDiGraph)
            edge_data = None
            if self.graph.has_edge(u, v):
                # Get first edge data
                edges = self.graph[u][v]
                if isinstance(edges, dict):
                    # MultiGraph - get first key
                    first_key = list(edges.keys())[0] if edges else 0
                    edge_data = edges[first_key]
                else:
                    edge_data = edges
            
            if not edge_data:
                print(f"      ⚠️  Warning: No edge data found for {u} -> {v}")
                continue
            
            u_coords = self.network.get_node_coords(u)
            v_coords = self.network.get_node_coords(v)
            
            length = edge_data.get('length', 0)
            
            # Fallback: calculate length from coordinates if missing
            if length == 0:
                length = self._haversine_distance(u_coords, v_coords)
            
            # Calculate travel time based on mode
            mode_speeds = {
                'walk': 1.4,      # 5 km/h
                'bike': 4.2,      # 15 km/h  
                'scooter': 5.6,   # 20 km/h
                'car': 13.9,      # 50 km/h
                'ev': 13.9,       # 50 km/h
                'drive': 13.9,    # 50 km/h (default)
           }

            default_speed = mode_speeds.get(mode, 13.9) 

            # Use OSM travel_time if available, otherwise calculate from mode
            if 'travel_time' in edge_data and edge_data['travel_time'] > 0:
                 travel_time = edge_data['travel_time']
            else:
                 travel_time = length / default_speed

            grade = edge_data.get('grade', 0)

            elevation_gain = max(0, length * grade) if grade else 0

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
            'route_id': f"route_{route_num}",
            'segments': segments,
            'total_distance_m': total_distance,
            'distance_km': total_distance / 1000,  # Add for convenience
            'estimated_travel_time_s': total_time,
            'eta_minutes': total_time / 60,  # Add for convenience
            'total_elevation_gain_m': total_elevation_gain,
            'node_path': path
        }
    
    @staticmethod
    def _haversine_distance(coord1: Tuple[float, float], 
                           coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two points."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
