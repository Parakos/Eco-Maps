# Eco Maps - Neural Route Prediction & Eco-Optimization

A live-updating, neural-route prediction and eco-optimizing routing application that runs entirely from the Ubuntu terminal.

## Features

- **Neural Route Prediction**: Small neural model predicting likely next route choices
- **Multi-Factor Scoring**: Composite environmental cost (energy, CO₂, time, safety)
- **Live Re-ranking**: Dynamic route updates based on context changes
- **Multi-Modal Support**: Walk, bike, scooter, car, EV, transit
- **Battery-Aware Routing**: Smart EV charging recommendations
- **Privacy-First**: Minimal data storage, opt-in telemetry (default off)

## System Requirements

- Ubuntu 20.04+ (or similar Linux distribution)
- Python 3.10+
- 4GB RAM minimum
- No GPU required

## Quick Setup

### 1. Automated Setup (Recommended)

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Train a Model

Train the neural route predictor on fixture data:

```bash
python -m eco_maps.cli ecotrain --config config/default_config.yml
```

This will:
- Load trajectory and route fixtures
- Train a compact LSTM-based model
- Save model weights to `data/models/route_predictor.pt`
- Display training metrics

### Start the Server

Launch the FastAPI server for API access:

```bash
python -m eco_maps.cli ecoserve start --port 8080
```

Server endpoints:
- `POST /predict` - Get route predictions with eco metrics
- `POST /re-rank` - Update rankings with new context

### Predict Routes (CLI)

Get route predictions and eco-scores from the command line:

```bash
python -m eco_maps.cli ecopredict \
  --user user_123 \
  --position "37.9838,23.7275" \
  --mode ev
```

Optional flags:
- `--mode`: walk, bike, scooter, car, ev, transit (default: walk)
- `--battery`: EV battery SOC percentage (default: 80)

Output example:
```json
{
  "predictions": [
    {
      "route_id": "route_001",
      "probability": 0.65,
      "eta_minutes": 12.5,
      "energy_kwh": 2.3,
      "co2_kg": 0.414,
      "monetary_cost": 0.46,
      "eco_score": 8.2,
      "requires_charging": false
    }
  ]
}
```

### Simulate Context Changes

Run scenarios showing live route re-ranking:

```bash
python -m eco_maps.cli ecosimulate \
  --scenario data/fixtures/sample_scenario.json
```

This demonstrates how routes are re-ranked when:
- Traffic conditions change
- Weather deteriorates
- Battery level drops
- Grid carbon intensity varies

## Configuration

Edit `config/default_config.yml` to customize:

```yaml
# User preferences
preferences:
  time_weight: 1.0
  co2_weight: 2.0
  monetary_weight: 0.5
  max_detour_pct: 20

# Model settings
model:
  sequence_length: 20
  hidden_size: 64
  num_layers: 2
  
# Training
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_scoring.py -v
pytest tests/test_model_infer.py -v
pytest tests/test_cli.py -v
```

## API Examples

### Predict Endpoint

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "position": {"lat": 37.9838, "lon": 23.7275},
    "mode": "ev",
    "battery_soc": 75,
    "context": {
      "weather": "clear",
      "time_of_day": "morning"
    }
  }'
```

### Re-rank Endpoint

```bash
curl -X POST http://localhost:8080/re-rank \
  -H "Content-Type: application/json" \
  -d '{
    "route_ids": ["route_001", "route_002"],
    "context_update": {
      "traffic_multiplier": 1.5,
      "battery_soc": 60
    }
  }'
```

## Project Structure

```
eco-maps/
├── README.md
├── setup.sh
├── requirements.txt
├── config/
│   └── default_config.yml
├── data/
│   ├── fixtures/
│   │   ├── trajectories.json
│   │   ├── candidate_routes.json
│   │   ├── weather_timeseries.json
│   │   ├── grid_carbon.json
│   │   └── sample_scenario.json
│   └── models/
├── eco_maps/
│   ├── __init__.py
│   ├── cli.py
│   ├── server.py
│   ├── model.py
│   ├── scoring.py
│   ├── data_loader.py
│   ├── candidate_gen.py
│   └── utils.py
├── tests/
│   ├── test_scoring.py
│   ├── test_model_infer.py
│   └── test_cli.py
└── examples/
    └── sample_runs.md
```

## Troubleshooting

### Model fails to train
- Ensure fixtures exist in `data/fixtures/`
- Check that you have enough disk space
- Verify Python version is 3.10+

### Server won't start
- Check if port 8080 is already in use
- Try a different port: `--port 8081`
- Verify all dependencies are installed

### Import errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## Privacy & Data

- **Minimal Storage**: Only essential identifiers stored
- **Opt-in Telemetry**: Disabled by default
- **Mock Data**: All examples use synthetic data
- **No External APIs**: All data sources are mockable fixtures

## Next Steps

See `developer_report.md` for:
- Architecture decisions
- Prototype limitations
- Integration with real data sources
- Production deployment considerations

## License

MIT License - See LICENSE file for details

## Support

For issues and questions, see `examples/sample_runs.md` for detailed walkthroughs.
