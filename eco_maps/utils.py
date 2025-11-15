"""Utility functions for Eco Maps."""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = "config/default_config.yml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_grid_carbon_intensity(timestamp: str, grid_data: list) -> float:
    """
    Get grid carbon intensity for a given timestamp.
    
    Args:
        timestamp: ISO format timestamp
        grid_data: List of grid carbon data points
    
    Returns:
        Carbon intensity in gCO2/kWh
    """
    from datetime import datetime
    
    ts = datetime.fromisoformat(timestamp.replace('+02:00', ''))
    
    # Find closest time point
    min_diff = float('inf')
    closest_intensity = 180  # Default
    
    for point in grid_data:
        point_ts = datetime.fromisoformat(point['time'].replace('+02:00', ''))
        diff = abs((ts - point_ts).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_intensity = point['gCO2_per_kWh']
    
    return closest_intensity
