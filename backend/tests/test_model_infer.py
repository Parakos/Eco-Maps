"""Tests for model inference."""

import pytest
import torch
from eco_maps.model import RoutePredictor, predict_routes


def test_model_forward_pass():
    """Test model forward pass with dummy data."""
    model = RoutePredictor(
        sequence_feature_dim=4,
        context_dim=5,
        hidden_size=32,
        num_layers=1,
        num_routes=10,
        dropout=0.0
    )
    
    # Dummy input
    sequence = torch.randn(2, 20, 4)  # [batch=2, seq_len=20, features=4]
    context = torch.randn(2, 5)  # [batch=2, context=5]
    
    output = model(sequence, context)
    
    assert output.shape == (2, 10)  # [batch, num_routes]


def test_predict_routes_output_format():
    """Test that predict_routes returns correct format."""
    model = RoutePredictor(
        sequence_feature_dim=4,
        context_dim=5,
        hidden_size=32,
        num_layers=1,
        num_routes=5,
        dropout=0.0
    )
    
    trajectory_seq = [
        {
            'lat': 37.98,
            'lon': 23.72,
            'speed_m_s': 5.0,
            'accel_m_s2': 0.1,
            'timestamp': '2025-11-14T08:00:00'
        }
        for _ in range(20)
    ]
    
    context = {
        'mode': 'walk',
        'hour': 8,
        'day': 1,
        'battery_soc': 80,
        'weather': 'clear'
    }
    
    routes = [
        {'route_id': f'route_{i:03d}', 'segments': []} for i in range(5)
    ]
    
    predictions = predict_routes(model, trajectory_seq, context, routes, top_k=3)
    
    assert len(predictions) <= 3
    assert all('route_id' in p for p in predictions)
    assert all('probability' in p for p in predictions)
    assert all(0 <= p['probability'] <= 1 for p in predictions)
