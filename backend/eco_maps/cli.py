"""
Command-line interface for Eco Maps.

Commands:
- ecotrain: Train the route prediction model
- ecopredict: Predict routes for a given position
- ecoserve: Start the API server
- ecosimulate: Simulate context changes and live re-ranking
"""
from pathlib import Path
from eco_maps import utils
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import time

from eco_maps import model, scoring, data_loader, server, utils


def cmd_train(args):
    """Train the route prediction model."""
    print("=" * 50)
    print("Eco Maps - Model Training")
    print("=" * 50)
    print()
    
    # Load config
    config = utils.load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Load data
    print("\nLoading training data...")
    trajectories = data_loader.load_trajectories()
    routes = data_loader.load_routes()
    
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Routes: {len(routes)}")
    
    # Train model
    print("\nStarting training...")
    trained_model = model.train_model(config, trajectories, routes)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    return 0


def cmd_predict(args):
    """Predict routes for a given position."""
    print("=" * 50)
    print("Eco Maps - Route Prediction")
    print("=" * 50)
    print()
    
    # Load config with environment variables
    config = utils.load_config(args.config)
    config = data_loader.load_config_with_env_vars(config)
    
    # Parse position
    try:
        lat, lon = map(float, args.position.split(','))
    except ValueError:
        print("Error: Position must be in format 'lat,lon'")
        return 1
    
    # Check if we need destination for OSM routing
    use_osm = config.get('external_apis', {}).get('use_osm_routing', False)
    
    if use_osm:
        if not hasattr(args, 'destination') or not args.destination:
            print("Error: --destination required when use_osm_routing is enabled")
            return 1
        try:
            dest_lat, dest_lon = map(float, args.destination.split(','))
        except ValueError:
            print("Error: Destination must be in format 'lat,lon'")
            return 1
    
    print(f"\nUser: {args.user}")
    print(f"Position: {lat}, {lon}")
    if use_osm:
        print(f"Destination: {dest_lat}, {dest_lon}")
    print(f"Mode: {args.mode}")
    if args.mode == 'ev':
        print(f"Battery SOC: {args.battery}%")
    
    # Load or generate routes
    if use_osm:
        print("\nGenerating routes from OpenStreetMap...")
        routes = data_loader.load_routes(
            use_osm=True,
            origin=(lat, lon),
            destination=(dest_lat, dest_lon),
            mode = args.mode
        )
        print(f"✓ Generated {len(routes)} routes")
    else:
        print("\nLoading fixture routes...")
        routes = data_loader.load_routes()
    
    # Get live context data
    print("Fetching live context data...")
    
    weather_data = data_loader.get_live_weather(lat, lon, config)
    print(f"  Weather: {weather_data.get('condition', 'unknown')}, "
          f"{weather_data.get('temp_c', 0):.1f}°C")
    
    if use_osm:
        traffic_data = data_loader.get_live_traffic(
            (lat, lon), (dest_lat, dest_lon), config
        )
        print(f"  Traffic: {traffic_data.get('traffic_level', 'unknown')} "
              f"({traffic_data.get('traffic_multiplier', 1.0):.1f}x)")
    else:
        traffic_data = {'traffic_multiplier': 1.0}
    
    grid_carbon = data_loader.get_live_grid_carbon(lat, lon, config)
    print(f"  Grid carbon: {grid_carbon:.0f} gCO2/kWh")
    
    # Create context
    from datetime import datetime
    now = datetime.now()
    context = {
        'mode': args.mode,
        'hour': now.hour,
        'day': now.weekday(),
        'battery_soc': args.battery,
        'weather': weather_data.get('condition', 'clear'),
        'traffic_multiplier': traffic_data.get('traffic_multiplier', 1.0),
        'grid_carbon_g_per_kwh': grid_carbon
    }
    
    # Score routes
    battery_model = None
    if args.mode == 'ev':
        battery_model = scoring.BatteryModel(
            capacity_kwh=60.0,
            soc_percent=args.battery,
            wh_per_km_base=150.0
        )
    
    print("\nScoring routes...")
    scored_routes = scoring.rank_routes(routes, context, config, battery_model)
    
    # Display results
    print("\n" + "=" * 50)
    print("Top Routes")
    print("=" * 50)
    
    for i, route in enumerate(scored_routes[:5], 1):
        print(f"\n{i}. Route {route['route_id']}")
        print(f"   ETA: {route['eta_minutes']:.1f} min")
        print(f"   Distance: {route['distance_km']:.2f} km")
        print(f"   Energy: {route['energy_kwh']:.3f} kWh")
        print(f"   CO2: {route['co2_kg']:.3f} kg")
        print(f"   Cost: ${route['monetary_cost']:.2f}")
        print(f"   Eco Score: {route['eco_score']:.2f}/10")
        if route.get('needs_charging'):
            print(f"   ⚠️  Charging required")
    
    return 0


def cmd_serve(args):
    """Start the API server."""
    print("=" * 50)
    print("Eco Maps - API Server")
    print("=" * 50)
    print()

    # 1. Load config to find the model's actual save path
    try:
        config = utils.load_config(args.config)
        # Use the save_path from the config file, which you correctly set to models/route_prediction_model.pt
        model_path = Path(config['model']['save_path']) 
    except Exception as e:
        # Fallback if config loading fails
        print(f"Warning: Could not load config to check model path: {e}")
        model_path = Path("data/models/route_predictor.pt")

    # 2. Check if the model exists using the correct path
    if not model_path.exists():
        print("Warning: Model not found. Some endpoints may not work.")
        print("Train the model first:")
        print(f"  python -m eco_maps.cli ecotrain --config {args.config}")
        print()

    print(f"Starting server on {args.host}:{args.port}...")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print()
    print("Press Ctrl+C to stop")
    print()

    # 3. Start server - FIXES THE TYPE ERROR (Removed the incorrect 'config_path' argument)
    server.run_server(host=args.host, port=args.port)

    return 0

def cmd_simulate(args):
    """Simulate context changes and show live re-ranking."""
    print("=" * 50)
    print("Eco Maps - Live Simulation")
    print("=" * 50)
    print()
    
    # Load model and config
    print("Loading model...")
    try:
        predictor, config = model.load_model()
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first.")
        return 1
    
    print("✓ Model loaded")
    
    # Load scenario
    print(f"\nLoading scenario: {args.scenario}")
    with open(args.scenario, 'r') as f:
        scenario = json.load(f)
    
    # Load routes
    routes = data_loader.load_routes()
    
    # Initial state
    pos = scenario['initial_position']
    mode = scenario['initial_mode']
    battery_soc = scenario.get('initial_battery_soc', 80)
    
    print(f"\nInitial State:")
    print(f"  Position: {pos['lat']}, {pos['lon']}")
    print(f"  Mode: {mode}")
    print(f"  Battery: {battery_soc}%")
    
    # Create mock trajectory
    trajectory_seq = []
    for i in range(20):
        trajectory_seq.append({
            'lat': pos['lat'] + i * 0.0001,
            'lon': pos['lon'] + i * 0.0001,
            'speed_m_s': 10.0,
            'accel_m_s2': 0.0,
            'timestamp': datetime.now().isoformat()
        })
    
    # Initial context
    context = {
        'mode': mode,
        'hour': 8,
        'day': 1,
        'battery_soc': battery_soc,
        'weather': 'clear',
        'traffic_multiplier': 1.0,
        'grid_carbon_g_per_kwh': 180
    }
    
    # Process events
    print("\n" + "=" * 50)
    print("Simulation Events")
    print("=" * 50)
    
    for event in scenario.get('events', []):
        time_offset = event.get('time_offset_s', 0)
        event_type = event['type']
        description = event.get('description', '')
        
        print(f"\n[T+{time_offset}s] {description}")
        
        # Update context based on event
        if event_type == 'traffic_update':
            context['traffic_multiplier'] = event.get('traffic_multiplier', 1.0)
        elif event_type == 'weather_change':
            context['weather'] = event.get('condition', 'clear')
        elif event_type == 'battery_update':
            context['battery_soc'] = event.get('battery_soc', battery_soc)
            battery_soc = context['battery_soc']
        
        # Re-predict and re-rank
        predictions = model.predict_routes(predictor, trajectory_seq, context, routes, top_k=3)
        
        # Score routes
        battery_model = None
        if mode == 'ev':
            battery_model = scoring.BatteryModel(
                capacity_kwh=60.0,
                soc_percent=battery_soc,
                wh_per_km_base=150.0
            )
        
        scored = []
        for pred in predictions:
            route = next((r for r in routes if r['route_id'] == pred['route_id']), None)
            if route:
                score = scoring.score_route(route, context, config, battery_model)
                score['probability'] = pred['probability']
                scored.append(score)
        
        # Display top 3
        print("  Updated Rankings:")
        for i, s in enumerate(scored[:3], 1):
            print(f"    {i}. {s['route_id']}: "
                  f"ETA={s['eta_minutes']:.1f}min, "
                  f"CO2={s['co2_kg']:.3f}kg, "
                  f"Score={s['eco_score']:.2f}")
        
        # Simulate time passing
        if args.realtime:
            time.sleep(2)
    
    print("\n" + "=" * 50)
    print("Simulation Complete")
    print("=" * 50)
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Eco Maps - Neural Route Prediction & Eco-Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ecotrain
    train_parser = subparsers.add_parser('ecotrain', help='Train the route prediction model')
    train_parser.add_argument('--config', default='config/default_config.yml',
                            help='Path to configuration file')
    
    # ecopredict
    predict_parser = subparsers.add_parser('ecopredict', help='Predict routes')
    predict_parser.add_argument('--destination',
                          help='Destination as "lat,lon" (required for OSM routing)')
    predict_parser.add_argument('--config', default='config/default_config.yml',
                          help='Path to configuration file')
    predict_parser.add_argument('--user', required=True, help='User ID')
    predict_parser.add_argument('--position', required=True,
                              help='Current position as "lat,lon"')
    predict_parser.add_argument('--mode', default='walk',
                              choices=['walk', 'bike', 'scooter', 'car', 'ev', 'transit'],
                              help='Transportation mode')
    predict_parser.add_argument('--battery', type=int, default=80,
                              help='Battery state of charge (for EV mode)')
    
    # ecoserve
    serve_parser = subparsers.add_parser('ecoserve', help='Start API server')
    serve_parser.add_argument('start', nargs='?', help='Start the server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    serve_parser.add_argument('--port', type=int, default=8080, help='Server port')
    serve_parser.add_argument('--config', default='config/default_config.yml', 
                              help='Path to config file')
    
    # ecosimulate
    simulate_parser = subparsers.add_parser('ecosimulate',
                                           help='Simulate context changes')
    simulate_parser.add_argument('--scenario', required=True,
                                help='Path to scenario JSON file')
    simulate_parser.add_argument('--realtime', action='store_true',
                                help='Run in real-time with delays')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    if args.command == 'ecotrain':
        return cmd_train(args)
    elif args.command == 'ecopredict':
        return cmd_predict(args)
    elif args.command == 'ecoserve':
        return cmd_serve(args)
    elif args.command == 'ecosimulate':
        return cmd_simulate(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
