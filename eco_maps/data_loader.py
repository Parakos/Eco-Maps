"""Data loading utilities for Eco Maps."""

import json
import re
import os
from typing import List, Dict
from pathlib import Path


def load_trajectories(path: str = "data/fixtures/trajectories.json") -> List[Dict]:
    """Load trajectory data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_routes(path: str = "data/fixtures/candidate_routes.json",
                use_osm: bool = False,
                origin: tuple = None,
                destination: tuple = None) -> List[Dict]:
    """
    Load routes - either from fixtures or generate from OSM.
    """
    if use_osm and origin and destination:
        # Generate routes dynamically from OSM
        from eco_maps.osm_loader import OSMRoadNetwork
        from eco_maps.path_planner import PathPlanner
        
        osm = OSMRoadNetwork()
        G = osm.load_area(center_point=origin, distance=10000)
        
        planner = PathPlanner(osm)
        routes = planner.find_k_shortest_paths(origin, destination, k=5)
        
        return routes
    else:
        # Load from fixtures (old behavior)
        with open(path, 'r') as f:
            return json.load(f)

def load_weather(path: str = "data/fixtures/weather_timeseries.json") -> List[Dict]:
    """Load weather time series from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_grid_carbon(path: str = "data/fixtures/grid_carbon.json") -> List[Dict]:
    """Load grid carbon intensity data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def load_config_with_env_vars(config: Dict) -> Dict:
    """Replace environment variable placeholders in config."""
    def replace_env_vars(value):
        if isinstance(value, str):
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            for match in matches:
                env_value = os.getenv(match, '')
                value = value.replace(f'${{{match}}}', env_value)
        elif isinstance(value, dict):
            return {k: replace_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_env_vars(v) for v in value]
        return value
    
    return replace_env_vars(config)


def get_live_weather(lat: float, lon: float, config: Dict) -> Dict:
    """Get weather data using configured source."""
    from eco_maps.adapters.weather import get_weather_source
    
    api_config = config.get('external_apis', {})
    source_type = api_config.get('weather_source', 'mock')
    api_key = api_config.get('openweathermap_api_key')
    
    source = get_weather_source(source_type, api_key)
    return source.get_weather(lat, lon)


def get_live_traffic(origin: tuple, destination: tuple, config: Dict) -> Dict:
    """Get traffic data using configured source."""
    from eco_maps.adapters.traffic import get_traffic_source
    
    api_config = config.get('external_apis', {})
    source_type = api_config.get('traffic_source', 'mock')
    
    # Get the right API key based on source
    if source_type == 'tomtom':
        api_key = api_config.get('tomtom_api_key')
    elif source_type == 'google':
        api_key = api_config.get('google_maps_api_key')
    else:
        api_key = None
    
    source = get_traffic_source(source_type, api_key)
    return source.get_traffic(origin, destination)


def get_live_grid_carbon(lat: float, lon: float, config: Dict) -> float:
    """Get grid carbon intensity using configured source."""
    from eco_maps.adapters.grid_carbon import get_grid_carbon_source
    
    api_config = config.get('external_apis', {})
    source_type = api_config.get('grid_carbon_source', 'mock')
    api_key = api_config.get('electricitymap_api_key')
    
    source = get_grid_carbon_source(source_type, api_key)
    return source.get_carbon_intensity(lat, lon)


def get_nearby_chargers(lat: float, lon: float, radius_km: float, 
                       config: Dict) -> List[Dict]:
    """Get nearby chargers using configured source."""
    from eco_maps.adapters.chargers import get_charger_source
    
    api_config = config.get('external_apis', {})
    source_type = api_config.get('charger_source', 'mock')
    api_key = api_config.get('openchargemap_api_key')
    
    source = get_charger_source(source_type, api_key)
    return source.find_chargers(lat, lon, radius_km)
