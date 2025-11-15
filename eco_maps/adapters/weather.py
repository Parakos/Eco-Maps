"""Weather data adapters for Eco Maps."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import requests
import json


class WeatherDataSource(ABC):
    """Abstract base class for weather data sources."""
    
    @abstractmethod
    def get_weather(self, lat: float, lon: float) -> Dict:
        pass


class MockWeatherSource(WeatherDataSource):
    """Mock weather source using fixtures."""
    
    def __init__(self, fixture_path: str = "data/fixtures/weather_timeseries.json"):
        with open(fixture_path, 'r') as f:
            self.data = json.load(f)
    
    def get_weather(self, lat: float, lon: float) -> Dict:
        if self.data:
            return self.data[0]
        return {
            'precip_mm': 0.0,
            'wind_m_s': 3.5,
            'temp_c': 15,
            'condition': 'clear'
        }


class OpenWeatherMapSource(WeatherDataSource):
    """Real weather from OpenWeatherMap API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, lat: float, lon: float) -> Dict:
        try:
            response = requests.get(self.base_url, params={
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            })
            response.raise_for_status()
            data = response.json()
            
            condition_map = {
                'Clear': 'clear',
                'Clouds': 'cloudy',
                'Rain': 'rain',
                'Drizzle': 'rain',
                'Snow': 'snow',
                'Thunderstorm': 'storm'
            }
            
            weather_main = data['weather'][0]['main']
            condition = condition_map.get(weather_main, 'clear')
            
            return {
                'temp_c': data['main']['temp'],
                'precip_mm': data.get('rain', {}).get('1h', 0.0),
                'wind_m_s': data['wind']['speed'],
                'condition': condition,
                'humidity': data['main']['humidity'],
                'source': 'openweathermap'
            }
            
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return MockWeatherSource().get_weather(lat, lon)


def get_weather_source(source_type: str = 'mock',
                      api_key: str = None) -> WeatherDataSource:
    """Factory function to get weather data source."""
    if source_type == 'mock':
        return MockWeatherSource()
    elif source_type == 'openweathermap':
        if not api_key:
            raise ValueError("API key required for OpenWeatherMap")
        return OpenWeatherMapSource(api_key)
    else:
        raise ValueError(f"Unknown weather source: {source_type}")
