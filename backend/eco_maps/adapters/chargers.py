"""EV charger data adapters for Eco Maps."""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import requests
import json


class ChargerDataSource(ABC):
    """Abstract base class for charger data sources."""
    
    @abstractmethod
    def find_chargers(self, lat: float, lon: float, radius_km: float = 10) -> List[Dict]:
        pass


class MockChargerSource(ChargerDataSource):
    """Mock charger source."""
    
    def find_chargers(self, lat: float, lon: float, radius_km: float = 10) -> List[Dict]:
        return [
            {
                'id': 'charger_001',
                'location': [lat + 0.01, lon + 0.01],
                'name': 'Mock Charger 1',
                'power_kw': 50,
                'available': True,
                'cost_per_kwh': 0.25
            }
        ]


class OpenChargeMapSource(ChargerDataSource):
    """Real charger data from OpenChargeMap API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.openchargemap.io/v3/poi"
    
    def find_chargers(self, lat: float, lon: float, radius_km: float = 10) -> List[Dict]:
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'distance': radius_km,
                'distanceunit': 'KM',
                'maxresults': 20
            }
            
            if self.api_key:
                params['key'] = self.api_key
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            chargers = []
            for station in data:
                address = station.get('AddressInfo', {})
                connections = station.get('Connections', [])
                
                if connections:
                    max_power = max([c.get('PowerKW', 0) for c in connections])
                else:
                    max_power = 0
                
                chargers.append({
                    'id': f"ocm_{station['ID']}",
                    'location': [address.get('Latitude'), address.get('Longitude')],
                    'name': address.get('Title', 'Unknown'),
                    'power_kw': max_power,
                    'available': station.get('StatusType', {}).get('IsOperational', False),
                    'cost_per_kwh': 0.20,  # Default, OCM doesn't always have pricing
                    'source': 'openchargemap'
                })
            
            return chargers
            
        except Exception as e:
            print(f"Error fetching chargers: {e}")
            return MockChargerSource().find_chargers(lat, lon, radius_km)


def get_charger_source(source_type: str = 'mock',
                      api_key: str = None) -> ChargerDataSource:
    """Factory function to get charger data source."""
    if source_type == 'mock':
        return MockChargerSource()
    elif source_type == 'openchargemap':
        return OpenChargeMapSource(api_key)
    else:
        raise ValueError(f"Unknown charger source: {source_type}")
