"""
Multi-factor route scoring and battery modeling for Eco Maps.

Computes composite environmental scores considering:
- Energy consumption
- CO2 emissions
- Travel time
- Safety factors
- Monetary cost
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math


class BatteryModel:
    """Simple battery model for electric vehicles."""
    
    def __init__(self, capacity_kwh: float = 60.0, soc_percent: float = 100.0,
                 wh_per_km_base: float = 150.0):
        """
        Args:
            capacity_kwh: Total battery capacity in kWh
            soc_percent: State of charge percentage (0-100)
            wh_per_km_base: Base energy consumption in Wh/km
        """
        self.capacity_kwh = capacity_kwh
        self.soc_percent = soc_percent
        self.wh_per_km_base = wh_per_km_base
    
    def available_energy_kwh(self) -> float:
        """Get available energy in kWh."""
        return self.capacity_kwh * (self.soc_percent / 100.0)
    
    def predict_consumption(self, route: Dict, context: Dict) -> float:
        """
        Predict energy consumption for a route.
        
        Args:
            route: Route with segments and attributes
            context: Context including weather, traffic, etc.
        
        Returns:
            Energy consumption in kWh
        """
        total_energy = 0.0
        
        for segment in route.get('segments', []):
            distance_km = segment['distance_m'] / 1000.0
            elevation_m = segment.get('elevation_gain_m', 0)
            
            # Base consumption
            base_wh = self.wh_per_km_base * distance_km
            
            # Elevation adjustment (10% increase per 10m climb)
            elevation_factor = 1.0 + (elevation_m / 100.0)
            
            # Speed adjustment (higher speeds = higher consumption)
            speed_m_s = context.get('avg_speed_m_s', 13.9)  # ~50 km/h default
            speed_factor = 0.8 + (speed_m_s / 50.0) * 0.4  # Range 0.8 to 1.2
            
            # Weather penalty (wind resistance, heating/cooling)
            weather = context.get('weather', 'clear')
            weather_factor = 1.2 if weather == 'rain' else 1.0
            
            # Traffic penalty (stop-and-go increases consumption)
            traffic_factor = context.get('traffic_multiplier', 1.0)
            
            segment_wh = base_wh * elevation_factor * speed_factor * weather_factor * traffic_factor
            total_energy += segment_wh / 1000.0  # Convert to kWh
        
        return total_energy
    
    def needs_charging(self, route_energy_kwh: float, min_soc: float = 15.0) -> bool:
        """
        Check if charging is needed for this route.
        
        Args:
            route_energy_kwh: Energy required for route
            min_soc: Minimum safe state of charge percentage
        
        Returns:
            True if charging needed
        """
        final_soc = self.soc_percent - (route_energy_kwh / self.capacity_kwh * 100.0)
        return final_soc < min_soc
    
    def update_soc(self, energy_used_kwh: float):
        """Update state of charge after energy consumption."""
        energy_pct = (energy_used_kwh / self.capacity_kwh) * 100.0
        self.soc_percent = max(0.0, self.soc_percent - energy_pct)


def calculate_co2_emissions(energy_kwh: float, mode: str, grid_carbon_g_per_kwh: float = 180,
                            fuel_co2_kg_per_liter: float = 2.31) -> float:
    """
    Calculate CO2 emissions for a route.
    
    Args:
        energy_kwh: Energy consumed
        mode: Transportation mode
        grid_carbon_g_per_kwh: Grid carbon intensity
        fuel_co2_kg_per_liter: CO2 per liter of fuel for ICE
    
    Returns:
        CO2 emissions in kg
    """
    if mode in ['walk', 'bike', 'transit']:
        return 0.0
    elif mode == 'ev' or mode == 'scooter':
        # Electric: use grid carbon intensity
        return (energy_kwh * grid_carbon_g_per_kwh) / 1000.0
    elif mode == 'car':
        # ICE: convert energy to fuel consumption
        # Assume ~8.5 kWh per liter of gasoline
        fuel_liters = energy_kwh / 8.5
        return fuel_liters * fuel_co2_kg_per_liter
    else:
        return 0.0


def calculate_monetary_cost(energy_kwh: float, mode: str, 
                           electricity_cost_per_kwh: float = 0.20,
                           fuel_cost_per_liter: float = 1.80) -> float:
    """
    Calculate monetary cost of route.
    
    Args:
        energy_kwh: Energy consumed
        mode: Transportation mode
        electricity_cost_per_kwh: Cost of electricity
        fuel_cost_per_liter: Cost of fuel
    
    Returns:
        Cost in currency units
    """
    if mode in ['walk', 'bike']:
        return 0.0
    elif mode == 'ev' or mode == 'scooter':
        return energy_kwh * electricity_cost_per_kwh
    elif mode == 'car':
        fuel_liters = energy_kwh / 8.5
        return fuel_liters * fuel_cost_per_liter
    elif mode == 'transit':
        return 2.5  # Fixed transit cost
    else:
        return 0.0


def calculate_safety_penalty(route: Dict, context: Dict) -> float:
    """
    Calculate safety penalty for route.
    
    Args:
        route: Route with segments
        context: Context including weather, time
    
    Returns:
        Safety penalty score (higher = less safe)
    """
    penalty = 0.0
    
    # Weather penalty
    weather = context.get('weather', 'clear')
    if weather == 'rain':
        penalty += 2.0
    elif weather == 'snow':
        penalty += 3.0
    
    # Night penalty for vulnerable modes
    hour = context.get('hour', 12)
    mode = route.get('mode', 'walk')
    if (hour < 6 or hour > 20) and mode in ['walk', 'bike', 'scooter']:
        penalty += 1.5
    
    # Road surface penalty
    for segment in route.get('segments', []):
        road_type = segment.get('road_type', 'paved')
        if road_type != 'paved':
            penalty += 1.0
        
        # Lack of bike lane for cycling
        if mode == 'bike' and not segment.get('has_bike_lane', False):
            penalty += 0.5
    
    return penalty


def score_route(route: Dict, context: Dict, config: Dict, 
                battery_model: Optional[BatteryModel] = None) -> Dict:
    """
    Score a route based on multiple environmental factors.
    
    Args:
        route: Route to score
        context: Current context (weather, traffic, etc.)
        config: Configuration with weights
        battery_model: Battery model for EVs
    
    Returns:
        Dictionary with detailed scoring breakdown
    """
    mode = route.get('mode', context.get('mode', 'walk'))
    
    # Calculate travel time
    eta_seconds = route.get('estimated_travel_time_s', 0)
    traffic_mult = context.get('traffic_multiplier', 1.0)
    eta_seconds *= traffic_mult
    eta_minutes = eta_seconds / 60.0
    
    # Calculate energy consumption
    energy_models = config.get('energy_models', {})
    base_wh_per_km = energy_models.get(mode, 0)
    
    if battery_model and mode == 'ev':
        energy_kwh = battery_model.predict_consumption(route, context)
        needs_charging = battery_model.needs_charging(energy_kwh)
    else:
        # Simple energy calculation
        total_distance_km = sum(s['distance_m'] for s in route.get('segments', [])) / 1000.0
        energy_kwh = (base_wh_per_km * total_distance_km) / 1000.0
        needs_charging = False
    
    # Calculate CO2 emissions
    grid_carbon = context.get('grid_carbon_g_per_kwh', 180)
    co2_kg = calculate_co2_emissions(energy_kwh, mode, grid_carbon)
    
    # Calculate monetary cost
    monetary_cost = calculate_monetary_cost(energy_kwh, mode)
    
    # Calculate safety penalty
    safety_penalty = calculate_safety_penalty(route, context)
    
    # Composite eco score using user preferences
    prefs = config.get('preferences', {})
    time_weight = prefs.get('time_weight', 1.0)
    co2_weight = prefs.get('co2_weight', 2.0)
    energy_weight = prefs.get('energy_weight', 1.5)
    monetary_weight = prefs.get('monetary_weight', 0.5)
    safety_weight = prefs.get('safety_weight', 1.0)
    
    # Negative because lower is better, negate to make higher score = better
    eco_score = -(
        time_weight * eta_minutes +
        co2_weight * co2_kg * 10 +  # Scale CO2
        energy_weight * energy_kwh +
        monetary_weight * monetary_cost +
        safety_weight * safety_penalty
    )
    
    # Normalize to 0-10 range (roughly)
    eco_score = max(0, min(10, eco_score / -10 + 10))
    
    return {
        'route_id': route.get('route_id', 'unknown'),
        'mode': mode,
        'eta_minutes': round(eta_minutes, 1),
        'energy_kwh': round(energy_kwh, 3),
        'co2_kg': round(co2_kg, 3),
        'monetary_cost': round(monetary_cost, 2),
        'safety_penalty': round(safety_penalty, 1),
        'eco_score': round(eco_score, 2),
        'needs_charging': needs_charging,
        'distance_km': round(sum(s['distance_m'] for s in route.get('segments', [])) / 1000.0, 2)
    }

def score_route_with_live_data(route: Dict, context: Dict, config: Dict,
                               battery_model: Optional[BatteryModel] = None,
                               chargers: List[Dict] = None) -> Dict:
    """
    Score route with live data integration.
    
    Args:
        route: Route to score
        context: Current context
        config: Configuration
        battery_model: Battery model for EVs
        chargers: List of available chargers
    
    Returns:
        Score dictionary with charger recommendations if needed
    """
    # Get base score
    score = score_route(route, context, config, battery_model)
    
    # Check if charging needed and add charger info
    if score.get('needs_charging') and chargers:
        from eco_maps.adapters.chargers import get_charger_source
        
        # Find chargers along route (simplified)
        route_center_lat = sum(s['start'][0] for s in route['segments']) / len(route['segments'])
        route_center_lon = sum(s['start'][1] for s in route['segments']) / len(route['segments'])
        
        # Get nearby chargers
        nearby = [c for c in chargers 
                 if abs(c['location'][0] - route_center_lat) < 0.05 
                 and abs(c['location'][1] - route_center_lon) < 0.05]
        
        if nearby:
            score['recommended_charger'] = nearby[0]
            # Add charging time to ETA
            charge_time_min = 20  # Assume 20 min fast charge
            score['eta_minutes'] += charge_time_min
    
    return score


def rank_routes(routes: List[Dict], context: Dict, config: Dict,
                battery_model: Optional[BatteryModel] = None) -> List[Dict]:
    """
    Score and rank multiple routes.
    
    Args:
        routes: List of candidate routes
        context: Current context
        config: Configuration
        battery_model: Battery model for EVs
    
    Returns:
        List of scored routes, sorted by eco_score (descending)
    """
    scored_routes = []
    
    for route in routes:
        score = score_route(route, context, config, battery_model)
        scored_routes.append(score)
    
    # Sort by eco_score (higher is better)
    scored_routes.sort(key=lambda x: x['eco_score'], reverse=True)
    
    return scored_routes


def find_charger_stops(route: Dict, battery_model: BatteryModel,
                      chargers: List[Dict], context: Dict) -> List[Dict]:
    """
    Find optimal charger stops for a route.
    
    Args:
        route: Route requiring charging
        battery_model: Current battery state
        chargers: Available chargers
        context: Current context
    
    Returns:
        List of recommended charger stops
    """
    # Mock implementation - in real system would use spatial index
    # For now, return first available charger
    if not chargers:
        return []
    
    # Simple heuristic: find charger near route midpoint
    charger_stops = []
    
    for charger in chargers[:1]:  # Just take first for prototype
        charge_time_minutes = 20  # Fast charge to 80%
        charger_stops.append({
            'charger_id': charger.get('id', 'charger_001'),
            'location': charger.get('location', [37.99, 23.73]),
            'charge_time_minutes': charge_time_minutes,
            'charge_kwh': 30,  # Partial charge
            'carbon_intensity': context.get('grid_carbon_g_per_kwh', 180)
        })
    
    return charger_stops
