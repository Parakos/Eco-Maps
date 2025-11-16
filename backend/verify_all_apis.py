#!/usr/bin/env python3
"""
Verify all API keys are working.
"""

import os
import requests
import json
from datetime import datetime


def test_here_maps():
    """Test HERE Maps routing API."""
    print("=" * 70)
    print("Testing HERE Maps API")
    print("=" * 70)
    
    api_key = os.getenv('HERE_API_KEY')
    if not api_key:
        print("❌ HERE_API_KEY not found")
        return False
    
    print(f"API Key: {api_key[:15]}...{api_key[-5:]}")
    
    url = "https://router.hereapi.com/v8/routes"
    params = {
        'apiKey': api_key,
        'transportMode': 'car',
        'origin': '37.9838,23.7275',
        'destination': '38.0,23.8',
        'return': 'summary,travelSummary'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return False
        
        data = response.json()
        
        if 'routes' in data:
            route = data['routes'][0]
            summary = route['sections'][0]['travelSummary']
            
            print("✅ HERE Maps API Working!")
            print(f"   Duration: {summary['duration']} seconds ({summary['duration']/60:.1f} min)")
            print(f"   Distance: {summary['length']} meters ({summary['length']/1000:.2f} km)")
            print(f"   Base Duration: {summary.get('baseDuration', 'N/A')} seconds")
            return True
        else:
            print(f"❌ Unexpected response: {json.dumps(data, indent=2)[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_openweathermap():
    """Test OpenWeatherMap API."""
    print("\n" + "=" * 70)
    print("Testing OpenWeatherMap API")
    print("=" * 70)
    
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    if not api_key:
        print("⚠️  OPENWEATHERMAP_API_KEY not found (optional)")
        return True
    
    print(f"API Key: {api_key[:15]}...{api_key[-5:]}")
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': 37.9838,
        'lon': 23.7275,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ API key not activated yet (wait 10-15 minutes)")
            return False
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return False
        
        data = response.json()
        
        if 'weather' in data and 'main' in data:
            print("✅ OpenWeatherMap API Working!")
            print(f"   Location: {data.get('name', 'Unknown')}, {data['sys']['country']}")
            print(f"   Weather: {data['weather'][0]['main']} - {data['weather'][0]['description']}")
            print(f"   Temperature: {data['main']['temp']:.1f}°C")
            print(f"   Humidity: {data['main']['humidity']}%")
            print(f"   Wind Speed: {data['wind']['speed']} m/s")
            return True
        else:
            print(f"❌ Unexpected response: {json.dumps(data, indent=2)[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_openchargemap():
    """Test OpenChargeMap API."""
    print("\n" + "=" * 70)
    print("Testing OpenChargeMap API")
    print("=" * 70)
    
    api_key = os.getenv('OPENCHARGEMAP_API_KEY')
    if not api_key:
        print("⚠️  OPENCHARGEMAP_API_KEY not found (API works without key, lower limits)")
    else:
        print(f"API Key: {api_key[:15]}...{api_key[-5:]}")
    
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        'latitude': 37.9838,
        'longitude': 23.7275,
        'distance': 5,
        'distanceunit': 'KM',
        'maxresults': 5
    }
    
    if api_key:
        params['key'] = api_key
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return False
        
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            print(f"✅ OpenChargeMap API Working!")
            print(f"   Found {len(data)} charging stations within 5 km")
            
            for i, station in enumerate(data[:3], 1):
                addr = station.get('AddressInfo', {})
                conns = station.get('Connections', [])
                
                print(f"\n   {i}. {addr.get('Title', 'Unknown')}")
                print(f"      Address: {addr.get('AddressLine1', 'N/A')}, {addr.get('Town', 'N/A')}")
                print(f"      Location: {addr.get('Latitude')}, {addr.get('Longitude')}")
                
                if conns:
                    power = max([c.get('PowerKW', 0) for c in conns])
                    print(f"      Max Power: {power} kW")
                    print(f"      Connectors: {len(conns)}")
            
            return True
        else:
            print("⚠️  No charging stations found (this might be normal)")
            print(f"   Response: {json.dumps(data, indent=2)[:200]}")
            return True  # Not necessarily an error
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all API tests."""
    print("\n" + "=" * 70)
    print("Eco Maps - API Verification Tool")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # Test each API
    results['HERE Maps'] = test_here_maps()
    results['OpenWeatherMap'] = test_openweathermap()
    results['OpenChargeMap'] = test_openchargemap()
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for api, passed in results.items():
        status = "✅ WORKING" if passed else "❌ FAILED"
        print(f"{api:20} {status}")
    
    print()
    
    if all(results.values()):
        print("🎉 All APIs are working correctly!")
        print("\nNext steps:")
        print("1. Test with CLI: python -m eco_maps.cli ecopredict --user test --position '37.98,23.72' --destination '38.0,23.8' --mode ev")
        print("2. Start server: python -m eco_maps.cli ecoserve start")
        return 0
    else:
        print("⚠️  Some APIs failed. Check errors above.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
