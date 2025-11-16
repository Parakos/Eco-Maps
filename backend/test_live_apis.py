"""Test live API connections."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_tomtom():
    """Test TomTom traffic API."""
    print("=" * 50)
    print("Testing TomTom Traffic API")
    print("=" * 50)
    
    try:
        from eco_maps.adapters.traffic import get_traffic_source
        
        api_key = os.getenv('TOMTOM_API_KEY')
        if not api_key:
            print("⚠️  TOMTOM_API_KEY not found in environment (skipping)")
            return True  # Not critical
        
        print(f"✓ API key loaded: {api_key[:20]}...")
        
        # Get traffic source
        source = get_traffic_source('tomtom', api_key)
        
        # Test with Athens coordinates
        origin = (37.9838, 23.7275)
        destination = (38.0000, 23.8000)
        
        print(f"Testing route: {origin} -> {destination}")
        
        traffic_data = source.get_traffic(origin, destination)
        
        # Check if we got real data or mock data
        if traffic_data['source'] == 'mock':
            print("\n⚠️  TomTom API returned mock data (API call failed)")
            print(f"{traffic_data}")
            return False
        
        print("\n✅ TomTom API Success!")
        print(f"{traffic_data}")
        print(f"Duration: {traffic_data['duration_in_traffic_s']} seconds")
        print(f"Traffic delay: {traffic_data.get('traffic_delay_s', 0)} seconds")
        print(f"Traffic level: {traffic_data['traffic_level']}")
        print(f"Distance: {traffic_data.get('distance_m', 'N/A')} meters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TomTom API Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_here_maps():
    """Test HERE Maps traffic API."""
    print("\n" + "=" * 50)
    print("Testing HERE Maps Traffic API")
    print("=" * 50)
    
    try:
        from eco_maps.adapters.traffic import get_traffic_source
        
        api_key = os.getenv('HERE_API_KEY')
        if not api_key:
            print("⚠️  HERE_API_KEY not found in environment (skipping)")
            return True  # Not critical
        
        print(f"✓ API key loaded: {api_key[:20]}...")
        
        # Get traffic source
        source = get_traffic_source('here', api_key)
        
        # Test with Athens coordinates
        origin = (37.9838, 23.7275)
        destination = (38.0000, 23.8000)
        
        print(f"Testing route: {origin} -> {destination}")
        
        traffic_data = source.get_traffic(origin, destination)
        
        # Check if we got real data or mock data
        if traffic_data['source'] == 'mock':
            print("\n⚠️  HERE Maps API returned mock data (API call failed)")
            print(f"{traffic_data}")
            return False
        
        print("\n✅ HERE Maps API Success!")
        print(f"{traffic_data}")
        print(f"Duration: {traffic_data['duration_in_traffic_s']} seconds")
        print(f"Traffic delay: {traffic_data.get('traffic_delay_s', 0)} seconds")
        print(f"Traffic level: {traffic_data['traffic_level']}")
        print(f"Distance: {traffic_data.get('distance_m', 'N/A')} meters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ HERE Maps API Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weather():
    """Test OpenWeatherMap API."""
    print("\n" + "=" * 50)
    print("Testing OpenWeatherMap API")
    print("=" * 50)
    
    try:
        from eco_maps.adapters.weather import get_weather_source
        
        api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        if not api_key:
            print("⚠️  OPENWEATHERMAP_API_KEY not found (optional)")
            return True  # Not critical
        
        print(f"✓ API key loaded: {api_key[:20]}...")
        
        # Get weather source
        source = get_weather_source('openweathermap', api_key)
        
        # Test with Athens
        lat, lon = 37.9838, 23.7275
        
        print(f"Testing weather for: {lat}, {lon} (Athens)")
        
        weather_data = source.get_weather(lat, lon)
        
        print("\n✅ Weather API Success!")
        print(f"Temperature: {weather_data.get('temp_c', 0):.1f}°C")
        print(f"Condition: {weather_data['condition']}")
        print(f"Wind: {weather_data.get('wind_m_s', 0):.1f} m/s")
        print(f"Humidity: {weather_data.get('humidity', 0)}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Weather API Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_carbon():
    """Test grid carbon (static)."""
    print("\n" + "=" * 50)
    print("Testing Grid Carbon (Static)")
    print("=" * 50)
    
    try:
        from eco_maps.adapters.grid_carbon import get_grid_carbon_source
        
        source = get_grid_carbon_source('static')
        
        # Test with Athens
        lat, lon = 37.9838, 23.7275
        
        print(f"Testing carbon intensity for: {lat}, {lon} (Athens)")
        
        intensity = source.get_carbon_intensity(lat, lon)
        
        print("\n✅ Grid Carbon Success!")
        print(f"Carbon intensity: {intensity:.0f} gCO2/kWh")
        print(f"Source: Static regional averages (Greece)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Grid Carbon Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Eco Maps - Live API Connection Test")
    print("=" * 50)
    print()
    
    results = {
        'TomTom': test_tomtom(),
        'HERE Maps': test_here_maps(),
        'Weather': test_weather(),
        'Grid Carbon': test_grid_carbon()
    }
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for api, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{api:15} {status}")
    
    print()
    
    # Check if at least one traffic API works
    traffic_works = results['TomTom'] or results['HERE Maps']
    
    if traffic_works and results['Grid Carbon']:
        print("🎉 Core APIs working! Ready to use live data.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check errors above.")
        print("💡 Tip: You need at least one traffic API (TomTom OR HERE Maps)")
        sys.exit(1)
