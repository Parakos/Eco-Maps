"""Tests for OSM data loader."""

import pytest
from eco_maps.osm_loader import OSMRoadNetwork


def test_load_small_area():
    """Test loading a small area."""
    osm = OSMRoadNetwork()
    
    # Load small area around Athens
    G = osm.load_area(
        center_point=(37.9838, 23.7275),
        distance=1000  # 1km radius
    )
    
    assert G is not None
    assert len(G.nodes) > 0
    assert len(G.edges) > 0


def test_nearest_node():
    """Test finding nearest node."""
    osm = OSMRoadNetwork()
    G = osm.load_area(center_point=(37.9838, 23.7275), distance=500)
    
    node = osm.get_nearest_node(37.9838, 23.7275)
    
    assert node is not None
    assert isinstance(node, (int, np.int64))


def test_caching():
    """Test that network is cached."""
    osm1 = OSMRoadNetwork()
    G1 = osm1.load_area(center_point=(37.9838, 23.7275), distance=500)
    
    # Second load should be from cache (faster)
    osm2 = OSMRoadNetwork()
    G2 = osm2.load_area(center_point=(37.9838, 23.7275), distance=500)
    
    assert len(G1.nodes) == len(G2.nodes)
