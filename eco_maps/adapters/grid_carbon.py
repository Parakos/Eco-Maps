# eco_maps/adapters/grid_carbon.py

from abc import ABC, abstractmethod
from typing import Dict
from datetime import datetime

class GridCarbonDataSource(ABC):
    @abstractmethod
    def get_carbon_intensity(self, lat: float, lon: float) -> float:
        pass


class MockGridCarbonSource(GridCarbonDataSource):
    """Mock source for testing."""
    def get_carbon_intensity(self, lat: float, lon: float) -> float:
        return 180.0


class StaticGridCarbonSource(GridCarbonDataSource):
    """Static regional averages with time-of-day adjustment."""
    
    # Average grid carbon intensity by country (gCO2/kWh)
    GRID_CARBON = {
        'GR': 380,  # Greece
        'DE': 385,  # Germany  
        'FR': 60,   # France
        'UK': 230,  # UK
        'US': 390,  # USA
        'IT': 350,  # Italy
        'ES': 220,  # Spain
    }
    
    # Time multipliers (renewables peak during day)
    TIME_FACTORS = {
        'night': 1.15,    # 0-6am
        'morning': 1.00,  # 6am-9am
        'midday': 0.85,   # 9am-3pm (solar peak)
        'evening': 1.10,  # 3pm-10pm (peak demand)
        'late': 1.15      # 10pm-midnight
    }
    
    def get_carbon_intensity(self, lat: float, lon: float) -> float:
        """Get carbon intensity with time-of-day adjustment."""
        country = self._detect_country(lat, lon)
        base_intensity = self.GRID_CARBON.get(country, 400)
        
        # Adjust for time of day
        hour = datetime.now().hour
        if 0 <= hour < 6:
            time_factor = self.TIME_FACTORS['night']
        elif 6 <= hour < 9:
            time_factor = self.TIME_FACTORS['morning']
        elif 9 <= hour < 15:
            time_factor = self.TIME_FACTORS['midday']
        elif 15 <= hour < 22:
            time_factor = self.TIME_FACTORS['evening']
        else:
            time_factor = self.TIME_FACTORS['late']
        
        return base_intensity * time_factor
    
    def _detect_country(self, lat: float, lon: float) -> str:
        """Detect country from coordinates."""
        # Greece
        if 34.8 <= lat <= 41.7 and 19.4 <= lon <= 28.2:
            return 'GR'
        
        # UK
        elif 49.9 <= lat <= 60.8 and -8.2 <= lon <= 1.8:
            return 'UK'
        
        # Germany
        elif 47.3 <= lat <= 55.0 and 5.9 <= lon <= 15.0:
            return 'DE'
        
        # France  
        elif 42.3 <= lat <= 51.1 and -5.0 <= lon <= 8.2:
            return 'FR'
        
        # Spain
        elif 36.0 <= lat <= 43.8 and -9.3 <= lon <= 3.3:
            return 'ES'
        
        # Italy
        elif 36.6 <= lat <= 47.1 and 6.6 <= lon <= 18.5:
            return 'IT'
        
        # USA (very rough)
        elif 24.5 <= lat <= 49.4 and -125.0 <= lon <= -66.9:
            return 'US'
        
        return 'UNKNOWN'


def get_grid_carbon_source(source_type: str = 'static',
                           api_key: str = None,
                           username: str = None,
                           password: str = None) -> GridCarbonDataSource:
    """Factory function to get grid carbon data source."""
    
    if source_type == 'mock':
        return MockGridCarbonSource()
    
    elif source_type == 'static':
        return StaticGridCarbonSource()
    
    elif source_type == 'watttime':
        if not username or not password:
            print("Warning: WattTime requires username/password, falling back to static")
            return StaticGridCarbonSource()
        # Would implement WattTimeSource here
        return StaticGridCarbonSource()
    
    elif source_type == 'electricitymap':
        print("Warning: ElectricityMap requires paid plan, falling back to static")
        return StaticGridCarbonSource()
    
    else:
        return StaticGridCarbonSource()
