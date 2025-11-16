"""Traffic data adapters for Eco Maps."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import googlemaps
from datetime import datetime
import json


class TrafficDataSource(ABC):
    """Abstract base class for traffic data sources."""
    
    @abstractmethod
    def get_traffic(self, origin: Tuple[float, float],
                   destination: Tuple[float, float],
                   mode: str = 'driving',
                   departure_time: Optional[datetime] = None) -> Dict:
        pass


class MockTrafficSource(TrafficDataSource):
    """Mock traffic source using fixtures."""
    
    def __init__(self, fixture_path: str = "data/fixtures/traffic.json"):
        self.fixture_path = fixture_path
        try:
            with open(fixture_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {'default_multiplier': 1.0}
    
    def get_traffic(self, origin, destination, mode='driving',
                   departure_time=None) -> Dict:
        return {
            'duration_in_traffic_s': 600,
            'traffic_multiplier': self.data.get('default_multiplier', 1.0),
            'traffic_level': 'moderate',
            'source': 'mock'
        }


class TomTomTrafficSource(TrafficDataSource):
    """Real traffic data from TomTom API (FREE tier available)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tomtom.com/routing/1/calculateRoute"
    
    def get_traffic(self, origin, destination, mode='car',
                   departure_time=None) -> Dict:
        """
        Get real-time traffic from TomTom.
        
        TomTom Free Tier: 2,500 requests/day
        """
        import requests
        
        # Format coordinates for TomTom
        route_points = f"{origin[0]},{origin[1]}:{destination[0]},{destination[1]}"
        
        # Map modes
        mode_map = {
            'driving': 'car',
            'car': 'car',
            'walk': 'pedestrian',
            'bike': 'bicycle',
            'ev': 'car'
        }
        travel_mode = mode_map.get(mode, 'car')
        
        params = {
            'key': self.api_key,
            'traffic': 'true',  # Include traffic
            'travelMode': travel_mode,
            'routeType': 'fastest',
            'computeBestOrder': 'false'
        }
        
        try:
            url = f"{self.base_url}/{route_points}/json"
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Debug: print response to see what we got
            if 'error' in data:
                print(f"TomTom API Error Response: {data['error']}")
                return MockTrafficSource().get_traffic(origin, destination)
            
            if 'routes' not in data or len(data['routes']) == 0:
                print(f"TomTom API returned no routes. Response: {data}")
                return MockTrafficSource().get_traffic(origin, destination)
            
            route = data['routes'][0]
            summary = route['summary']
            
            # TomTom provides traffic delay in seconds
            duration_traffic = summary['travelTimeInSeconds']
            traffic_delay = summary.get('trafficDelayInSeconds', 0)
            duration_normal = duration_traffic - traffic_delay
            
            if duration_normal > 0:
                traffic_multiplier = duration_traffic / duration_normal
            else:
                traffic_multiplier = 1.0
            
            # Determine traffic level
            if traffic_multiplier < 1.2:
                level = 'light'
            elif traffic_multiplier < 1.5:
                level = 'moderate'
            elif traffic_multiplier < 2.0:
                level = 'heavy'
            else:
                level = 'severe'
            
            return {
                'duration_in_traffic_s': duration_traffic,
                'duration_normal_s': duration_normal,
                'traffic_multiplier': traffic_multiplier,
                'traffic_level': level,
                'traffic_delay_s': traffic_delay,
                'distance_m': summary['lengthInMeters'],
                'source': 'tomtom'
            }
            
        except requests.exceptions.HTTPError as e:
            print(f"TomTom HTTP Error: {e}")
            print(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            return MockTrafficSource().get_traffic(origin, destination)
        except Exception as e:
            print(f"Error fetching traffic from TomTom: {e}")
            import traceback
            traceback.print_exc()
            return MockTrafficSource().get_traffic(origin, destination)


class HEREMapsTrafficSource(TrafficDataSource):
    """Real traffic data from HERE Maps API (FREE tier available)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://router.hereapi.com/v8/routes"
    
    def get_traffic(self, origin, destination, mode='car',
                   departure_time=None) -> Dict:
        """
        Get real-time traffic from HERE Maps.
        
        HERE Free Tier: 250,000 requests/month
        """
        import requests
        
        # Map modes
        mode_map = {
            'driving': 'car',
            'car': 'car',
            'walk': 'pedestrian',
            'bike': 'bicycle',
            'ev': 'car'
        }
        transport_mode = mode_map.get(mode, 'car')
        
        params = {
            'apiKey': self.api_key,
            'transportMode': transport_mode,
            'origin': f"{origin[0]},{origin[1]}",
            'destination': f"{destination[0]},{destination[1]}",
            'return': 'summary',
            'departureTime': departure_time.isoformat() if departure_time else 'any'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Debug: print response to see what we got
            if 'error' in data:
                print(f"HERE Maps API Error Response: {data}")
                return MockTrafficSource().get_traffic(origin, destination)
            
            if 'routes' not in data or len(data['routes']) == 0:
                print(f"HERE Maps returned no routes. Response: {data}")
                return MockTrafficSource().get_traffic(origin, destination)
            
            route = data['routes'][0]
            sections = route['sections'][0]
            summary = sections['summary']
            
            # HERE provides duration with and without traffic
            duration_traffic = summary['duration']  # in seconds, includes traffic
            base_duration = summary.get('baseDuration', duration_traffic)
            
            if base_duration > 0:
                traffic_multiplier = duration_traffic / base_duration
            else:
                traffic_multiplier = 1.0
            
            traffic_delay = duration_traffic - base_duration
            
            # Determine traffic level
            if traffic_multiplier < 1.2:
                level = 'light'
            elif traffic_multiplier < 1.5:
                level = 'moderate'
            elif traffic_multiplier < 2.0:
                level = 'heavy'
            else:
                level = 'severe'
            
            return {
                'duration_in_traffic_s': duration_traffic,
                'duration_normal_s': base_duration,
                'traffic_multiplier': traffic_multiplier,
                'traffic_level': level,
                'traffic_delay_s': traffic_delay,
                'distance_m': summary['length'],
                'source': 'here_maps'
            }
            
        except requests.exceptions.HTTPError as e:
            print(f"HERE Maps HTTP Error: {e}")
            print(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            return MockTrafficSource().get_traffic(origin, destination)
        except Exception as e:
            print(f"Error fetching traffic from HERE Maps: {e}")
            import traceback
            traceback.print_exc()
            return MockTrafficSource().get_traffic(origin, destination)


class GoogleMapsTrafficSource(TrafficDataSource):
    """Real traffic data from Google Maps (PAID - keeping for reference)."""
    
    def __init__(self, api_key: str):
        self.client = googlemaps.Client(key=api_key)
    
    def get_traffic(self, origin, destination, mode='driving',
                   departure_time=None) -> Dict:
        if departure_time is None:
            departure_time = 'now'
        
        try:
            result = self.client.distance_matrix(
                origins=[origin],
                destinations=[destination],
                mode=mode,
                departure_time=departure_time,
                traffic_model='best_guess'
            )
            
            element = result['rows'][0]['elements'][0]
            
            if element['status'] != 'OK':
                return MockTrafficSource().get_traffic(origin, destination)
            
            duration_traffic = element.get('duration_in_traffic', {}).get('value')
            duration_normal = element['duration']['value']
            
            if duration_traffic:
                traffic_multiplier = duration_traffic / duration_normal
            else:
                traffic_multiplier = 1.0
            
            if traffic_multiplier < 1.2:
                level = 'light'
            elif traffic_multiplier < 1.5:
                level = 'moderate'
            elif traffic_multiplier < 2.0:
                level = 'heavy'
            else:
                level = 'severe'
            
            return {
                'duration_in_traffic_s': duration_traffic or duration_normal,
                'duration_normal_s': duration_normal,
                'traffic_multiplier': traffic_multiplier,
                'traffic_level': level,
                'distance_m': element['distance']['value'],
                'source': 'google_maps'
            }
            
        except Exception as e:
            print(f"Error fetching traffic: {e}")
            return MockTrafficSource().get_traffic(origin, destination)


def get_traffic_source(source_type: str = 'mock',
                      api_key: Optional[str] = None) -> TrafficDataSource:
    """Factory function to get traffic data source."""
    if source_type == 'mock':
        return MockTrafficSource()
    elif source_type == 'tomtom':
        if not api_key:
            raise ValueError("API key required for TomTom")
        return TomTomTrafficSource(api_key)
    elif source_type == 'here':
        if not api_key:
            raise ValueError("API key required for HERE Maps")
        return HEREMapsTrafficSource(api_key)
    elif source_type == 'google':
        if not api_key:
            raise ValueError("API key required for Google Maps")
        return GoogleMapsTrafficSource(api_key)
    else:
        raise ValueError(f"Unknown traffic source: {source_type}")
