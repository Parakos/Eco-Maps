"""Tests for route scoring and battery models."""

import pytest
from eco_maps.scoring import (
    BatteryModel, calculate_co2_emissions, calculate_monetary_cost,
    calculate_safety_penalty, score_route, rank_routes
)


def test_battery_model_consumption():
    """Test battery energy consumption prediction."""
    battery = BatteryModel(capacity_kwh=60.0, soc_percent=80.0, wh_per_km_base=150.0)
    
    route = {
        'segments': [
            {
                'distance_m': 10000,  # 10 km
                'elevation_gain_m': 50
            }
        ]
    }
    
    context = {
        'avg_speed_m_s': 13.9,
        'weather': 'clear',
        'traffic_multiplier': 1.0
    }
    
    energy = battery.predict_consumption(route, context)
    
    # Should be around 1.5 kWh for 10km
    assert 1.0 < energy < 3.0
    assert isinstance(energy, float)


def test_battery_charging_detection():
    """Test that charging is needed when SOC is low."""
    battery = BatteryModel(capacity_kwh=60.0, soc_percent=20.0, wh_per_km_base=150.0)
    
    # Route requiring 15 kWh
    needs_charging = battery.needs_charging(route_energy_kwh=15.0, min_soc=15.0)
    
    # 20% of 60kWh = 12kWh available, need 15kWh
    assert needs_charging is True


def test_co2_calculation_ev():
    """Test CO2 calculation for electric vehicles."""
    co2 = calculate_co2_emissions(
        energy_kwh=10.0,
        mode='ev',
        grid_carbon_g_per_kwh=200
    )
    
    # 10 kWh * 200 g/kWh = 2000g = 2kg
    assert abs(co2 - 2.0) < 0.01


def test_co2_calculation_walk():
    """Test CO2 calculation for walking (should be zero)."""
    co2 = calculate_co2_emissions(
        energy_kwh=0.0,
        mode='walk',
        grid_carbon_g_per_kwh=200
    )
    
    assert co2 == 0.0


def test_scoring_downhill_vs_uphill():
    """Test that downhill routes score better than uphill."""
    config = {
        'preferences': {
            'time_weight': 1.0,
            'co2_weight': 2.0,
            'energy_weight': 1.5,
            'monetary_weight': 0.5,
            'safety_weight': 1.0
        },
        'energy_models': {
            'bike': 0
        }
    }
    
    route_downhill = {
        'route_id': 'downhill',
        'mode': 'bike',
        'segments': [
            {'distance_m': 5000, 'elevation_gain_m': -20, 'road_type': 'paved', 'has_bike_lane': True}
        ],
        'estimated_travel_time_s': 600
    }
    
    route_uphill = {
        'route_id': 'uphill',
        'mode': 'bike',
        'segments': [
            {'distance_m': 5000, 'elevation_gain_m': 50, 'road_type': 'paved', 'has_bike_lane': True}
        ],
        'estimated_travel_time_s': 900
    }
    
    context = {
        'mode': 'bike',
        'weather': 'clear',
        'hour': 12,
        'traffic_multiplier': 1.0,
        'grid_carbon_g_per_kwh': 180
    }
    
    score_down = score_route(route_downhill, context, config)
    score_up = score_route(route_uphill, context, config)
    
    # Downhill should have better (higher) eco score
    assert score_down['eco_score'] > score_up['eco_score']
    assert score_down['eta_minutes'] < score_up['eta_minutes']


def test_safety_penalty_rain():
    """Test that rain increases safety penalty."""
    route = {
        'route_id': 'test',
        'mode': 'bike',
        'segments': [
            {'distance_m': 1000, 'road_type': 'paved', 'has_bike_lane': True}
        ]
    }
    
    context_clear = {'weather': 'clear', 'hour': 12}
    context_rain = {'weather': 'rain', 'hour': 12}
    
    penalty_clear = calculate_safety_penalty(route, context_clear)
    penalty_rain = calculate_safety_penalty(route, context_rain)
    
    assert penalty_rain > penalty_clear


def test_preference_weights_affect_ranking():
    """Test that CO2 weight influences route ranking."""
    routes = [
        {
            'route_id': 'low_co2',
            'mode': 'ev',
            'segments': [
                {'distance_m': 3000, 'elevation_gain_m': 0, 'road_type': 'paved'}
            ],
            'estimated_travel_time_s': 600
        },
        {
            'route_id': 'high_co2',
            'mode': 'car',
            'segments': [
                {'distance_m': 2500, 'elevation_gain_m': 0, 'road_type': 'paved'}
            ],
            'estimated_travel_time_s': 500
        }
    ]
    
    context = {
        'mode': 'ev',
        'weather': 'clear',
        'hour': 12,
        'traffic_multiplier': 1.0,
        'grid_carbon_g_per_kwh': 180
    }
    
    # High CO2 weight
    config_eco = {
        'preferences': {
            'time_weight': 0.5,
            'co2_weight': 5.0,  # Very high
            'energy_weight': 1.0,
            'monetary_weight': 0.5,
            'safety_weight': 1.0
        },
        'energy_models': {
            'ev': 150,
            'car': 180
        }
    }
    
    ranked = rank_routes(routes, context, config_eco)
    
    # Low CO2 route should rank higher with high CO2 weight
    assert ranked[0]['route_id'] == 'low_co2'
