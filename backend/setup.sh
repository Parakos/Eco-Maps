#!/bin/bash

set -e

echo "========================================="
echo "Eco Maps - Automated Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Create virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf venv
fi

echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/fixtures
mkdir -p data/models
mkdir -p config
mkdir -p tests
mkdir -p examples
echo "✓ Directories created"
echo ""

# Generate fixture data if not exists
echo "Checking for fixture data..."
if [ ! -f "data/fixtures/trajectories.json" ]; then
    echo "Generating sample fixtures..."
    python3 -c "
import json
import os
from datetime import datetime, timedelta

# Create fixtures directory
os.makedirs('data/fixtures', exist_ok=True)

# Generate trajectories
trajectories = []
base_time = datetime(2025, 11, 14, 8, 0, 0)
for i in range(100):
    trajectories.append({
        'user_id': f'user_{i % 10}',
        'timestamp': (base_time + timedelta(minutes=i)).isoformat(),
        'lat': 37.9838 + (i % 10) * 0.001,
        'lon': 23.7275 + (i % 10) * 0.001,
        'speed_m_s': 1.4 + (i % 5) * 0.2,
        'accel_m_s2': 0.1,
        'mode': ['walk', 'bike', 'ev'][i % 3]
    })

with open('data/fixtures/trajectories.json', 'w') as f:
    json.dump(trajectories, f, indent=2)

# Generate candidate routes
routes = []
for i in range(20):
    routes.append({
        'route_id': f'route_{i:03d}',
        'segments': [
            {
                'start': [37.98 + i*0.001, 23.72 + i*0.001],
                'end': [37.99 + i*0.001, 23.73 + i*0.001],
                'distance_m': 1000 + i*100,
                'elevation_gain_m': 10 + i*2,
                'road_type': 'paved',
                'has_bike_lane': i % 2 == 0
            }
        ],
        'mode': ['walk', 'bike', 'ev', 'car'][i % 4],
        'estimated_travel_time_s': 600 + i*60
    })

with open('data/fixtures/candidate_routes.json', 'w') as f:
    json.dump(routes, f, indent=2)

# Generate weather data
weather = []
for i in range(24):
    weather.append({
        'timestamp': (base_time + timedelta(hours=i)).isoformat(),
        'precip_mm': 0.0 if i % 4 != 0 else 2.5,
        'wind_m_s': 3.5 + (i % 3),
        'temp_c': 10 + i % 8,
        'condition': 'clear' if i % 4 != 0 else 'rain'
    })

with open('data/fixtures/weather_timeseries.json', 'w') as f:
    json.dump(weather, f, indent=2)

# Generate grid carbon data
grid_carbon = []
for i in range(24):
    grid_carbon.append({
        'time': (base_time + timedelta(hours=i)).isoformat(),
        'gCO2_per_kWh': 150 + (i % 12) * 10
    })

with open('data/fixtures/grid_carbon.json', 'w') as f:
    json.dump(grid_carbon, f, indent=2)

# Generate sample scenario
scenario = {
    'initial_position': {'lat': 37.9838, 'lon': 23.7275},
    'initial_mode': 'ev',
    'initial_battery_soc': 80,
    'events': [
        {
            'time_offset_s': 0,
            'type': 'start',
            'description': 'Journey begins'
        },
        {
            'time_offset_s': 300,
            'type': 'traffic_update',
            'affected_routes': ['route_001', 'route_002'],
            'traffic_multiplier': 1.5,
            'description': 'Traffic spike on main routes'
        },
        {
            'time_offset_s': 600,
            'type': 'weather_change',
            'condition': 'rain',
            'description': 'Rain begins'
        },
        {
            'time_offset_s': 900,
            'type': 'battery_update',
            'battery_soc': 60,
            'description': 'Battery depleted to 60%'
        }
    ]
}

with open('data/fixtures/sample_scenario.json', 'w') as f:
    json.dump(scenario, f, indent=2)

print('✓ Sample fixtures generated')
"
    echo "✓ Sample fixtures generated"
else
    echo "✓ Fixtures already exist"
fi
echo ""

# Create default config if not exists
if [ ! -f "config/default_config.yml" ]; then
    echo "Creating default configuration..."
    cat > config/default_config.yml << 'EOF'
# Eco Maps Configuration

# User preferences for route scoring
preferences:
  time_weight: 1.0          # Weight for travel time
  co2_weight: 2.0           # Weight for CO2 emissions
  monetary_weight: 0.5      # Weight for monetary cost
  energy_weight: 1.5        # Weight for energy consumption
  safety_weight: 1.0        # Weight for safety factors
  max_detour_pct: 20        # Maximum acceptable detour percentage

# Model architecture settings
model:
  sequence_length: 20       # Number of historical points to consider
  hidden_size: 64           # LSTM hidden dimension
  num_layers: 2             # Number of LSTM layers
  context_size: 16          # Context vector dimension
  dropout: 0.2              # Dropout rate

# Training parameters
training:
  epochs: 50                # Training epochs
  batch_size: 32            # Batch size
  learning_rate: 0.001      # Learning rate
  validation_split: 0.2     # Validation data split
  
# Energy models (Wh/km baseline)
energy_models:
  walk: 0                   # No energy consumption
  bike: 0                   # No energy consumption
  scooter: 15               # Electric scooter
  car: 180                  # ICE vehicle
  ev: 150                   # Electric vehicle
  transit: 0                # Handled separately

# Battery model defaults
battery:
  capacity_kwh: 60          # Default EV battery capacity
  min_soc_pct: 15           # Minimum safe state of charge
  charge_rate_kw: 50        # Fast charger rate

# Safety penalties
safety:
  rain_penalty: 2.0         # Penalty multiplier for rain
  night_penalty: 1.5        # Penalty multiplier for night
  poor_surface_penalty: 1.3 # Penalty for unpaved roads

# Server settings
server:
  host: "0.0.0.0"
  port: 8080
  debug: false

# Privacy settings
privacy:
  telemetry_enabled: false  # Opt-in telemetry
  store_trajectories: false # Store user trajectories
  anonymize_user_ids: true  # Hash user identifiers
EOF
    echo "✓ Default configuration created"
else
    echo "✓ Configuration already exists"
fi
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Train the model:"
echo "   python -m eco_maps.cli ecotrain --config config/default_config.yml"
echo ""
echo "3. Start the server:"
echo "   python -m eco_maps.cli ecoserve start"
echo ""
echo "4. Run predictions:"
echo "   python -m eco_maps.cli ecopredict --user user_123 --position '37.9838,23.7275' --mode ev"
echo ""
echo "5. Run tests:"
echo "   pytest tests/ -v"
echo ""
echo "See README.md for more information."
echo ""
