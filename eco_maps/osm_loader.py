"""
OpenStreetMap data loader for Eco Maps.
"""

import osmnx as ox
import networkx as nx
from typing import Tuple, Dict, List, Optional
import pickle
from pathlib import Path


class OSMRoadNetwork:
    """Load and manage OpenStreetMap road network."""
    
    def __init__(self, cache_dir: str = "data/osm_cache"):
        """
        Initialize OSM road network loader.
        
        Args:
            cache_dir: Directory to cache downloaded networks
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.graph = None
        
    def load_area(self, place_name: str = None, 
                  center_point: Tuple[float, float] = None,
                  distance: int = 5000) -> nx.MultiDiGraph:
        """
        Load road network for an area.
        
        Args:
            place_name: Name of place (e.g., "Athens, Greece")
            center_point: (lat, lon) tuple for center
            distance: Radius in meters from center
            
        Returns:
            NetworkX graph of road network
        """
        cache_file = self.cache_dir / f"{place_name or 'custom'}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            print(f"Loading cached network from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
            return self.graph
        
        # Download from OSM
        print("Downloading road network from OpenStreetMap...")
        
        if place_name:
            self.graph = ox.graph_from_place(
                place_name,
                network_type='all',
                simplify=True
            )
        elif center_point:
            self.graph = ox.graph_from_point(
                center_point,
                dist=distance,
                network_type='all',
                simplify=True
            )
        else:
            raise ValueError("Must provide place_name or center_point")
        
        # Add edge attributes for routing
        self.graph = ox.add_edge_speeds(self.graph)
        self.graph = ox.add_edge_travel_times(self.graph)
        
        # Add elevation data (optional)
        try:
            self.graph = ox.add_node_elevations_google(
                self.graph, 
                api_key=None  # Uses free SRTM data if None
            )
            self.graph = ox.add_edge_grades(self.graph)
        except Exception as e:
            print(f"Warning: Could not add elevation data: {e}")
        
        # Cache for future use
        with open(cache_file, 'wb') as f:
            pickle.dump(self.graph, f)
        
        print(f"Network loaded: {len(self.graph.nodes)} nodes, "
              f"{len(self.graph.edges)} edges")
        
        return self.graph
    
    def get_nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest network node to a coordinate."""
        if self.graph is None:
            raise ValueError("Network not loaded. Call load_area() first.")
        
        return ox.nearest_nodes(self.graph, lon, lat)
    
    def get_node_coords(self, node_id: int) -> Tuple[float, float]:
        """Get (lat, lon) coordinates of a node."""
        node = self.graph.nodes[node_id]
        return (node['y'], node['x'])
    
    def get_edge_attributes(self, u: int, v: int, key: int = 0) -> Dict:
        """Get attributes of an edge."""
        return self.graph.edges[u, v, key]
