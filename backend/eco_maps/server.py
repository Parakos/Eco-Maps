"""FastAPI server for Eco Maps."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

from eco_maps import model, scoring, data_loader, utils


app = FastAPI(
    title="Eco Maps API",
    description="Neural route prediction and eco-optimization",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    # This allows your React app running on localhost:5173 to access the API.
    # If your frontend runs on a different port, change this to match.
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"], # Allows all headers (like Content-Type)
)

# Global state
_model = None
_config = None
_routes = None


class PredictRequest(BaseModel):
    """Request model for route prediction."""
    user_id: str
    position: Dict[str, float]  # {lat, lon}
    mode: str = "walk"
    battery_soc: Optional[int] = 80
    context: Optional[Dict] = {}


class ReRankRequest(BaseModel):
    """Request model for route re-ranking."""
    route_ids: List[str]
    context_update: Dict


@app.on_event("startup")
async def startup():
    """Load model and data on startup."""
    global _model, _config, _routes
    
    try:
        _model, _config = model.load_model()
        _routes = data_loader.load_routes()
        print("✓ Model and data loaded")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        _model = None
        _config = utils.load_config()
        _routes = data_loader.load_routes()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Eco Maps API",
        "version": "0.1.0",
        "status": "running",
        "model_loaded": _model is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": _model is not None
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Predict optimal routes with live data."""
    if _model is None and not _config.get('external_apis', {}).get('use_osm_routing'):
        raise HTTPException(status_code=503, detail="Model not loaded and OSM routing disabled")
    
    # Get live context data
    lat, lon = request.position['lat'], request.position['lon']
    
    # Fetch live weather
    weather_data = data_loader.get_live_weather(lat, lon, _config)
    
    # Fetch grid carbon
    grid_carbon = data_loader.get_live_grid_carbon(lat, lon, _config)
    
    # Build context
    now = datetime.now()
    context = {
        'mode': request.mode,
        'hour': now.hour,
        'day': now.weekday(),
        'battery_soc': request.battery_soc,
        'weather': weather_data.get('condition', 'clear'),
        'traffic_multiplier': request.context.get('traffic_multiplier', 1.0),
        'grid_carbon_g_per_kwh': grid_carbon
    }
    
    # Get routes
    use_osm = _config.get('external_apis', {}).get('use_osm_routing', False)
    
    if use_osm and request.context.get('destination'):
        dest = request.context['destination']
        routes = data_loader.load_routes(
            use_osm=True,
            origin=(lat, lon),
            destination=(dest['lat'], dest['lon'])
        )
    else:
        routes = _routes
    
    # Get traffic for each route
    if request.context.get('destination'):
        dest = request.context['destination']
        traffic_data = data_loader.get_live_traffic(
            (lat, lon),
            (dest['lat'], dest['lon']),
            _config
        )
        context['traffic_multiplier'] = traffic_data.get('traffic_multiplier', 1.0)
    
    # Get chargers if EV mode
    chargers = None
    if request.mode == 'ev':
        chargers = data_loader.get_nearby_chargers(lat, lon, 10, _config)
    
    # Score routes
    battery_model = None
    if request.mode == 'ev':
        battery_model = scoring.BatteryModel(
            capacity_kwh=60.0,
            soc_percent=request.battery_soc,
            wh_per_km_base=150.0
        )
    
    scored = []
    for route in routes[:10]:  # Limit to top 10
        score = scoring.score_route_with_live_data(
            route, context, _config, battery_model, chargers
        )
        scored.append(score)
    
    # Sort by eco score
    scored.sort(key=lambda x: x.get('eco_score', -999999), reverse=True)
    
    return {
        'user_id': request.user_id,
        'timestamp': datetime.now().isoformat(),
        'position': request.position,
        'mode': request.mode,
        'context': context,
        'predictions': scored[:5]
    }

@app.post("/re-rank")
async def re_rank(request: ReRankRequest):
    """
    Re-rank routes with updated context.
    
    Useful for live updates when traffic, weather, or battery changes.
    """
    # Filter routes
    routes_to_rank = [r for r in _routes if r['route_id'] in request.route_ids]
    
    if not routes_to_rank:
        raise HTTPException(status_code=404, detail="No matching routes found")
    
    # Build context
    now = datetime.now()
    context = {
        'mode': request.context_update.get('mode', 'walk'),
        'hour': now.hour,
        'day': now.weekday(),
        'battery_soc': request.context_update.get('battery_soc', 80),
        'weather': request.context_update.get('weather', 'clear'),
        'traffic_multiplier': request.context_update.get('traffic_multiplier', 1.0),
        'grid_carbon_g_per_kwh': request.context_update.get('grid_carbon_g_per_kwh', 180)
    }
    
    # Score and rank
    scored = scoring.rank_routes(routes_to_rank, context, _config)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'context': request.context_update,
        'ranked_routes': scored
    }


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)
