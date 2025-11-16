"""Candidate route generation for Eco Maps."""

from typing import List, Dict, Tuple
import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points.
    
    Returns distance in meters.
    """
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def filter_relevant_routes(routes: List[Dict], position: Tuple[float, float],
                          destination: Tuple[float, float] = None,
                          max_distance_m: float = 10000) -> List[Dict]:
    """
    Filter routes relevant to current position.
    
    Args:
        routes: All available routes
        position: Current (lat, lon)
        destination: Optional destination (lat, lon)
        max_distance_m: Maximum distance to consider
    
    Returns:
        Filtered list of relevant routes
    """
    relevant = []
    
    for route in routes:
        # Check if route starts near current position
        if route.get('segments'):
            start = route['segments'][0]['start']
            dist = haversine_distance(position[0], position[1], start[0], start[1])
            
            if dist <= max_distance_m:
                relevant.append(route)
    
    return relevant


def generate_multimodal_routes(routes: List[Dict], modes: List[str]) -> List[Dict]:
    """
    Generate multi-modal route combinations.
    
    Args:
        routes: Base routes
        modes: List of modes to combine
    
    Returns:
        List of multi-modal routes
    """
    # Simplified: just return single-mode routes for prototype
    # Real implementation would combine route segments across modes
    return routes
