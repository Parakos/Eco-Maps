"""
Neural route prediction model for Eco Maps.

Architecture:
- LSTM for trajectory sequence processing
- MLP for context encoding
- Fusion layer combining sequence and context
- Output layer predicting route probabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class RouteDataset(Dataset):
    """Dataset for trajectory sequences and route choices."""
    
    def __init__(self, trajectories: List[Dict], routes: List[Dict], 
                 sequence_length: int = 20):
        """
        Args:
            trajectories: List of trajectory points
            routes: List of available routes
            sequence_length: Number of points in sequence
        """
        self.sequence_length = sequence_length
        self.routes = {r['route_id']: i for i, r in enumerate(routes)}
        self.samples = self._create_samples(trajectories)
        
    def _create_samples(self, trajectories: List[Dict]) -> List[Dict]:
        """Create training samples from trajectories."""
        samples = []
        user_trajs = {}
        
        # Group by user
        for traj in trajectories:
            uid = traj['user_id']
            if uid not in user_trajs:
                user_trajs[uid] = []
            user_trajs[uid].append(traj)
        
        # Create sequences
        for uid, trajs in user_trajs.items():
            trajs = sorted(trajs, key=lambda x: x['timestamp'])
            for i in range(len(trajs) - self.sequence_length):
                seq = trajs[i:i+self.sequence_length]
                # Mock: assign a route based on pattern
                route_idx = hash(uid + str(i)) % len(self.routes)
                samples.append({
                    'sequence': seq,
                    'route_idx': route_idx,
                    'context': self._extract_context(seq[-1])
                })
        
        return samples
    
    def _extract_context(self, point: Dict) -> np.ndarray:
        """Extract context features from a trajectory point."""
        mode_map = {'walk': 0, 'bike': 1, 'scooter': 2, 'car': 3, 'ev': 4, 'transit': 5}
        mode = mode_map.get(point.get('mode', 'walk'), 0)
        
        # Extract hour and day features
        from datetime import datetime
        ts = datetime.fromisoformat(point['timestamp'].replace('+02:00', ''))
        hour = ts.hour / 24.0
        day = ts.weekday() / 7.0
        
        context = np.array([
            mode / 5.0,  # Normalized mode
            hour,  # Time of day
            day,  # Day of week
            0.8,  # Mock battery level
            0.0,  # Mock weather (clear)
        ], dtype=np.float32)
        
        return context
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract sequence features: [lat, lon, speed, accel]
        seq_features = np.array([
            [p['lat'], p['lon'], p['speed_m_s'], p.get('accel_m_s2', 0.0)]
            for p in sample['sequence']
        ], dtype=np.float32)
        
        # Normalize
        seq_features[:, 0] = (seq_features[:, 0] - 37.98) * 100  # Lat
        seq_features[:, 1] = (seq_features[:, 1] - 23.72) * 100  # Lon
        seq_features[:, 2] = seq_features[:, 2] / 10.0  # Speed
        seq_features[:, 3] = seq_features[:, 3] / 5.0  # Accel
        
        return (
            torch.FloatTensor(seq_features),
            torch.FloatTensor(sample['context']),
            torch.LongTensor([sample['route_idx']])
        )


class RoutePredictor(nn.Module):
    """Neural model for route prediction."""
    
    def __init__(self, sequence_feature_dim: int = 4, context_dim: int = 5,
                 hidden_size: int = 64, num_layers: int = 2, 
                 num_routes: int = 20, dropout: float = 0.2):
        """
        Args:
            sequence_feature_dim: Dimension of sequence features
            context_dim: Dimension of context vector
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_routes: Number of output routes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Sequence encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=sequence_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Context encoder (MLP)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_routes)
        )
    
    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [batch, seq_len, feature_dim]
            context: [batch, context_dim]
        
        Returns:
            Route logits [batch, num_routes]
        """
        # Process sequence
        lstm_out, _ = self.lstm(sequence)
        seq_encoding = lstm_out[:, -1, :]  # Take last hidden state
        
        # Process context
        ctx_encoding = self.context_encoder(context)
        
        # Fuse and predict
        fused = torch.cat([seq_encoding, ctx_encoding], dim=1)
        logits = self.fusion(fused)
        
        return logits


def train_model(config: Dict, trajectories: List[Dict], routes: List[Dict],
                save_path: str = "data/models/route_predictor.pt") -> RoutePredictor:
    """
    Train the route prediction model.
    
    Args:
        config: Configuration dictionary
        trajectories: Training trajectories
        routes: Available routes
        save_path: Path to save model
    
    Returns:
        Trained model
    """
    print("Preparing dataset...")
    dataset = RouteDataset(
        trajectories, 
        routes,
        sequence_length=config['model']['sequence_length']
    )
    
    # Split train/val
    val_size = int(len(dataset) * config['training']['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Initialize model
    model = RoutePredictor(
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_routes=len(routes),
        dropout=config['model']['dropout']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    print(f"\nTraining for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for seq, ctx, target in train_loader:
            optimizer.zero_grad()
            output = model(seq, ctx)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_correct += (predicted == target.squeeze()).sum().item()
            train_total += target.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for seq, ctx, target in val_loader:
                output = model(seq, ctx)
                loss = criterion(output, target.squeeze())
                
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == target.squeeze()).sum().item()
                val_total += target.size(0)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['training']['epochs']}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Acc: {100*train_correct/train_total:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Acc: {100*val_correct/val_total:.2f}%")
    
    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'num_routes': len(routes)
    }, save_path)
    
    print(f"\n✓ Model saved to {save_path}")
    
    return model


def load_model(model_path: str = "data/models/route_predictor.pt") -> Tuple[RoutePredictor, Dict]:
    """
    Load a trained model.
    
    Args:
        model_path: Path to saved model

    Returns:
        (model, config) tuple
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    model = RoutePredictor(
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_routes=checkpoint['num_routes'],
        dropout=config['model']['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def predict_routes(model: RoutePredictor, trajectory_seq: List[Dict],
                   context: Dict, routes: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Predict top-K routes given trajectory and context.
    
    Args:
        model: Trained model
        trajectory_seq: Recent trajectory points
        context: Current context (mode, battery, weather, etc.)
        routes: Available routes
        top_k: Number of top routes to return
    
    Returns:
        List of predictions with route_id and probability
    """
    # Prepare sequence
    seq_features = np.array([
        [p['lat'], p['lon'], p['speed_m_s'], p.get('accel_m_s2', 0.0)]
        for p in trajectory_seq[-20:]  # Last 20 points
    ], dtype=np.float32)
    
    # Normalize
    seq_features[:, 0] = (seq_features[:, 0] - 37.98) * 100
    seq_features[:, 1] = (seq_features[:, 1] - 23.72) * 100
    seq_features[:, 2] = seq_features[:, 2] / 10.0
    seq_features[:, 3] = seq_features[:, 3] / 5.0
    
    # Pad if needed
    if len(seq_features) < 20:
        pad = np.zeros((20 - len(seq_features), 4), dtype=np.float32)
        seq_features = np.vstack([pad, seq_features])
    
    # Prepare context
    mode_map = {'walk': 0, 'bike': 1, 'scooter': 2, 'car': 3, 'ev': 4, 'transit': 5}
    mode_val = mode_map.get(context.get('mode', 'walk'), 0)
    
    ctx_features = np.array([
        mode_val / 5.0,
        context.get('hour', 12) / 24.0,
        context.get('day', 0) / 7.0,
        context.get('battery_soc', 80) / 100.0,
        1.0 if context.get('weather') == 'rain' else 0.0
    ], dtype=np.float32)
    
    # Predict
    with torch.no_grad():
        seq_tensor = torch.FloatTensor(seq_features).unsqueeze(0)
        ctx_tensor = torch.FloatTensor(ctx_features).unsqueeze(0)
        logits = model(seq_tensor, ctx_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    
    # Get top-K
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        if idx < len(routes):
            predictions.append({
                'route_id': routes[idx]['route_id'],
                'probability': float(probs[idx])
            })
    
    return predictions
