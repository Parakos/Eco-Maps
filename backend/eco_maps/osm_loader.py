"""
OpenStreetMap data loader for Eco Maps.
"""

import osmnx as ox
import networkx as nx
from typing import Tuple, Dict, List, Optional
import pickle
from pathlib import Path
import os


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
                  distance: int = 5000,
                  network_type: str = 'drive') -> nx.MultiDiGraph:
        """
        Load road network for an area.
        
        Args:
            place_name: Name of place (e.g., "Athens, Greece")
            center_point: (lat, lon) tuple for center
            distance: Radius in meters from center
            network_type: Type of street network ('drive', 'walk', 'bike', 'all')
            
        Returns:
            NetworkX graph of road network
        """
        # Use GraphML format instead of pickle (much faster!)
        cache_name = place_name or 'custom'
        # Sanitize filename
        cache_name = cache_name.replace(',', '').replace(' ', '_').lower()
        cache_file = self.cache_dir / f"{cache_name}.graphml"
        
        # Try to load from cache
        if cache_file.exists():
            print(f"Loading cached network from {cache_file}")
            
            # Check file size
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"Cache file size: {file_size_mb:.1f} MB")
            
            try:
                # GraphML loads MUCH faster than pickle
                self.graph = ox.load_graphml(cache_file)
                
                print(f"✓ Cache loaded successfully: {len(self.graph.nodes)} nodes, "
                      f"{len(self.graph.edges)} edges")
                return self.graph
                
            except Exception as e:
                print(f"⚠️  Cache loading failed: {e}")
                print("Regenerating network from OSM...")
                try:
                    cache_file.unlink()  # Delete corrupted cache
                except:
                    pass
        
        # Download from OSM
        print("Downloading road network from OpenStreetMap...")
        
        try:
            if place_name:
                self.graph = ox.graph_from_place(
                    place_name,
                    network_type=network_type,
                    simplify=True
                )
            elif center_point:
                self.graph = ox.graph_from_point(
                    center_point,
                    dist=distance,
                    network_type=network_type,
                    simplify=True
                )
            else:
                raise ValueError("Must provide place_name or center_point")
        
        except Exception as e:
            print(f"❌ Error downloading from OSM: {e}")
            raise
        
        # Add edge attributes for routing
        print("Adding routing attributes...")
        self.graph = ox.add_edge_speeds(self.graph)
        self.graph = ox.add_edge_travel_times(self.graph)
        
        # Add elevation data (optional)
        try:
            print("Attempting to add elevation data...")
            google_key = os.getenv('GOOGLE_ELEVATION_API_KEY')
            
            if google_key:
                self.graph = ox.add_node_elevations_google(
                    self.graph, 
                    api_key=google_key
                )
                self.graph = ox.add_edge_grades(self.graph)
                print("✓ Elevation data added")
            else:
                # Try SRTM free data
                self.graph = ox.add_node_elevations_google(
                    self.graph, 
                    api_key=None
                )
                self.graph = ox.add_edge_grades(self.graph)
                print("✓ Elevation data added (SRTM)")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not add elevation data: {e}")
            print("Continuing without elevation data...")
        
        # Save to cache using GraphML (much faster than pickle!)
        try:
            print("Saving network to cache...")
            ox.save_graphml(self.graph, cache_file)
            print(f"✓ Cache saved to {cache_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save cache: {e}")
            # Continue anyway - cache is optional
        
        print(f"✓ Network loaded: {len(self.graph.nodes)} nodes, "
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
    
    def clear_cache(self):
        """Clear all cached network files."""
        print("Clearing OSM cache...")
        count = 0
        for cache_file in self.cache_dir.glob("*.graphml"):
            cache_file.unlink()
            count += 1
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        print(f"✓ Deleted {count} cache files")
