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
        # Set a class attribute for the number of available routes
        self.num_routes = len(self.routes)
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
            
            # 1. SEQUENCE LENGTH CHECK
            if len(trajs) < 2:
                print(f"DEBUG FILTER: Skipping user {uid}. Only {len(trajs)} points found (requires >= 2).")
                continue

            # The sequence is the history (all but the last point)
            sequence = trajs[:-1]
            # The last point is the prediction point, containing the label
            traj = trajs[-1] 
            chosen_route_id = traj.get('chosen_route_id')

            # 2. CHOSEN ROUTE ID VALIDATION
            if chosen_route_id is None:
                print(f"DEBUG FILTER: Skipping user {uid}. Last point is missing 'chosen_route_id'.")
                continue

            if chosen_route_id not in self.routes:
                # This is the most likely failure point based on history!
                print(f"DEBUG FILTER: Skipping user {uid}. Route ID '{chosen_route_id}' not found in candidate routes.")
                print(f"DEBUG INFO: Available route keys: {list(self.routes.keys())[:5]}... Total: {len(self.routes)}")
                continue

            # 3. CONTEXT FIELD VALIDATION (Used for feature engineering)
            context = traj.get('context')
            if context is None:
                print(f"DEBUG FILTER: Skipping user {uid}. Last point is missing 'context' dictionary.")
                continue

            # If all checks pass, create the sample
            samples.append({
                'sequence': sequence,
                'context': context,
                'label': self.routes[chosen_route_id]
            })
            
            print(f"DEBUG SUCCESS: Created sample for {uid} with label for route {chosen_route_id}")


        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Sequence features
        seq_features = RoutePredictionModel._get_sequence_features(sample['sequence'])
        
        # Context features
        ctx_features = RoutePredictionModel._get_context_features(sample['context'])
        
        # Label
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return seq_features, ctx_features, label


class RoutePredictionModel(nn.Module):
    """LSTM-based model for predicting user route choice."""

    def __init__(self, config: Dict, num_routes: int):
        super().__init__()
        
        self.hidden_dim = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        
        # 4 trajectory features: lat, lon, speed_m_s, accel_m_s2
        self.lstm = nn.LSTM(input_size=4, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_layers, 
                            batch_first=True)
        
        # 5 context features: mode, hour, day, battery_soc, weather_rain
        self.context_mlp = nn.Sequential(
            nn.Linear(5, self.hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Fusion layer: LSTM output (hidden_dim) + Context MLP output (hidden_dim/2)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(),
        )
        
        # Output layer: Predict probabilities over all routes
        self.output_layer = nn.Linear(self.hidden_dim, num_routes)

    def forward(self, seq_features: torch.Tensor, ctx_features: torch.Tensor) -> torch.Tensor:
        
        # 1. Process Sequence (Trajectory)
        # Output shape: (batch_size, sequence_length, hidden_dim)
        # h_n, c_n shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(seq_features)
        
        # Use the hidden state of the last layer, last time step
        # h_n[-1] has shape (batch_size, hidden_dim)
        sequence_vector = h_n[-1] 
        
        # 2. Process Context
        context_vector = self.context_mlp(ctx_features) # shape: (batch_size, hidden_dim/2)
        
        # 3. Fusion
        fused_vector = torch.cat((sequence_vector, context_vector), dim=1) # shape: (batch_size, hidden_dim + hidden_dim/2)
        fused_output = self.fusion_mlp(fused_vector)
        
        # 4. Output (Logits)
        logits = self.output_layer(fused_output)
        
        return logits
    
    @staticmethod
    def _get_sequence_features(sequence: List[Dict], sequence_length: int = 20) -> torch.Tensor:
        """
        Extracts and normalizes features from a trajectory sequence.
        
        Features: lat, lon, speed_m_s, accel_m_s2
        """
        # Feature extraction
        # Use .get() with a default value to prevent KeyErrors on potentially bad data
        seq_features = np.array([
            [
                traj.get('lat', 0.0), 
                traj.get('lon', 0.0), 
                traj.get('speed_m_s', 0.0), 
                traj.get('accel_m_s2', 0.0)
            ]
            for traj in sequence
        ], dtype=np.float32)
        
        if len(seq_features) == 0:
            # Handle empty sequence after filtering, though _create_samples should prevent this
            seq_features = np.zeros((0, 4), dtype=np.float32)

        # Normalize (assuming data is centered around 37.98, 23.72)
        if len(seq_features) > 0:
            seq_features[:, 0] = (seq_features[:, 0] - 37.98) * 100
            seq_features[:, 1] = (seq_features[:, 1] - 23.72) * 100
            seq_features[:, 2] = seq_features[:, 2] / 10.0 # Speed normalization
            seq_features[:, 3] = seq_features[:, 3] / 5.0  # Accel normalization
        
        # Pad or truncate to fixed sequence_length (pad at the beginning)
        if len(seq_features) < sequence_length:
            pad = np.zeros((sequence_length - len(seq_features), 4), dtype=np.float32)
            seq_features = np.vstack([pad, seq_features])
        elif len(seq_features) > sequence_length:
            seq_features = seq_features[-sequence_length:]
            
        return torch.tensor(seq_features, dtype=torch.float32)

    @staticmethod
    def _get_context_features(context: Dict) -> torch.Tensor:
        """
        Extracts and normalizes features from a context dictionary.
        
        Features: mode, hour, day, battery_soc, weather
        """
        mode_map = {'walk': 0, 'bike': 1, 'scooter': 2, 'car': 3, 'ev': 4, 'transit': 5}
        mode_val = mode_map.get(context.get('mode', 'walk'), 0)
        
        ctx_features = np.array([
            mode_val / 5.0, # 0.0 to 1.0 (6 modes, max is 5)
            context.get('hour', 12) / 24.0, # 0.0 to 1.0
            context.get('day', 0) / 7.0, # 0.0 to 1.0
            context.get('battery_soc', 80) / 100.0, # 0.0 to 1.0
            1.0 if context.get('weather') == 'rain' else 0.0 # Binary flag
        ], dtype=np.float32)
        
        return torch.tensor(ctx_features, dtype=torch.float32)


def train_model(config: Dict, trajectories: List[Dict], routes: List[Dict]) -> RoutePredictionModel:
    """Trains the route prediction model."""
    
    print("Preparing dataset...")
    # Initialize the dataset
    dataset = RouteDataset(trajectories, routes)
    
    if len(dataset) == 0:
        # Re-raise the error with a more helpful message
        raise ValueError(
            f"num_samples should be a positive integer value, but got num_samples=0. "
            f"This means ALL {len(trajectories)} trajectories were filtered out. "
            f"Check the DEBUG FILTER messages above to see why!"
        )

    # Initialize model, optimizer, and loss function
    model = RoutePredictionModel(config, dataset.num_routes)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    # Training loop
    epochs = config['training']['epochs']
    print(f"Training on {len(dataset)} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for seq_features, ctx_features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seq_features, ctx_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Save model
    save_path = Path(config['model']['save_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model


def load_model(config_path: str = "config/default_config.yml") -> Tuple[RoutePredictionModel, Dict]:
    """Loads the model and configuration."""
    from eco_maps import data_loader, utils # Import here to prevent circular dependency issues
    
    config = utils.load_config(config_path)
    
    # We need the number of routes to initialize the final layer
    # Load a small fixture file of routes just to get the count
    try:
        routes_fixture = data_loader.load_routes()
        num_routes = len(routes_fixture)
    except Exception as e:
        print(f"Warning: Could not load routes fixture to determine model size: {e}")
        # Default to 20 routes if fixture loading fails
        num_routes = 20 

    model = RoutePredictionModel(config, num_routes)
    
    # Load state dictionary
    save_path = Path(config['model']['save_path'])
    if not save_path.exists():
        raise FileNotFoundError(f"Model file not found at {save_path}. Run 'ecotrain' first.")
        
    model.load_state_dict(torch.load(save_path))
    model.eval() # Set to evaluation mode
    
    return model, config


def predict_route(model: RoutePredictionModel, config: Dict, 
                  trajectories: List[Dict], 
                  available_routes: List[Dict]) -> List[Tuple[str, float]]:
    """
    Predicts the best route based on the last trajectory point and context.
    
    Returns: List of (route_id, probability) tuples.
    """
    if not trajectories:
        return []

    # 1. Prepare Features
    # The last trajectory point should contain the context for prediction
    traj = trajectories[-1]
    
    # Get sequence features (history)
    # Use the full history for prediction, minus the current point (if available)
    sequence = trajectories[:-1] if len(trajectories) > 1 else [] 

    seq_features = RoutePredictionModel._get_sequence_features(sequence, sequence_length=config['model']['sequence_length'])
    
    # Get context features from the current point
    context = traj.get('context', {}) # Use .get() for safety
    ctx_features = RoutePredictionModel._get_context_features(context)
    
    # Add batch dimension (size 1)
    seq_features = seq_features.unsqueeze(0)
    ctx_features = ctx_features.unsqueeze(0)
    
    # 2. Predict
    with torch.no_grad():
        logits = model(seq_features, ctx_features)
        probabilities = torch.softmax(logits, dim=1)
        
    # 3. Format Output
    # Get the index->route_id map
    route_map = {i: r['route_id'] for i, r in enumerate(available_routes)}
    
    # Extract top probabilities
    probs, indices = torch.topk(probabilities, k=min(5, len(available_routes)))
    
    predictions = []
    for prob, index in zip(probs.squeeze().tolist(), indices.squeeze().tolist()):
        route_id = route_map.get(index)
        if route_id:
            predictions.append((route_id, prob))
            
    return predictions
