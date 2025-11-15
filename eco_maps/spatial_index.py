"""
Spatial indexing for fast geographic queries.
"""

from rtree import index
from typing import List, Tuple, Dict
import numpy as np


class SpatialRouteIndex:
    """R-tree spatial index for routes."""
    
    def __init__(self):
        """Initialize spatial index."""
        self.idx = index.Index()
        self.routes = {}
        self.route_counter = 0
    
    def insert_route(self, route: Dict):
        """
        Insert a route into the spatial index.
        
        Args:
            route: Route dictionary with segments
        """
        route_id = route['route_id']
        
        # Calculate bounding box
        all_coords = []
        for segment in route['segments']:
            all_coords.append(segment['start'])
            all_coords.append(segment['end'])
        
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        
        # Bounding box: (min_lon, min_lat, max_lon, max_lat)
        bbox = (min(lons), min(lats), max(lons), max(lats))
        
        # Insert into R-tree
        self.idx.insert(self.route_counter, bbox)
        self.routes[self.route_counter] = route
        self.route_counter += 1
    
    def find_nearby_routes(self, point: Tuple[float, float],
                          radius_m: float = 1000) -> List[Dict]:
        """
        Find routes near a point.
        
        Args:
            point: (lat, lon)
            radius_m: Search radius in meters
            
        Returns:
            List of nearby routes
        """
        lat, lon = point
        
        # Convert radius to degrees (rough approximation)
        radius_deg = radius_m / 111000
        
        # Query bounding box
        bbox = (
            lon - radius_deg,
            lat - radius_deg,
            lon + radius_deg,
            lat + radius_deg
        )
        
        # Find intersecting routes
        nearby_ids = list(self.idx.intersection(bbox))
        
        nearby_routes = [self.routes[rid] for rid in nearby_ids]
        
        return nearby_routes
    
    def find_routes_between(self, origin: Tuple[float, float],
                           destination: Tuple[float, float]) -> List[Dict]:
        """
        Find routes that connect origin and destination.
        
        Args:
            origin: (lat, lon)
            destination: (lat, lon)
            
        Returns:
            Candidate routes
        """
        # Find routes near origin
        origin_routes = self.find_nearby_routes(origin, radius_m=500)
        
        # Find routes near destination
        dest_routes = self.find_nearby_routes(destination, radius_m=500)
        
        # Find intersection
        origin_ids = {r['route_id'] for r in origin_routes}
        dest_ids = {r['route_id'] for r in dest_routes}
        
        connecting_ids = origin_ids & dest_ids
        
        connecting_routes = [
            r for r in origin_routes if r['route_id'] in connecting_ids
        ]
        
        return connecting_routes


# Example usage
if __name__ == "__main__":
    # Create index
    idx = SpatialRouteIndex()
    
    # Insert some routes
    route1 = {
        'route_id': 'route_001',
        'segments': [
            {'start': [37.98, 23.72], 'end': [37.99, 23.73]},
            {'start': [37.99, 23.73], 'end': [38.00, 23.74]}
        ]
    }
    
    idx.insert_route(route1)
    
    # Query
    nearby = idx.find_nearby_routes((37.985, 23.725), radius_m=1000)
    print(f"Found {len(nearby)} nearby routes")
