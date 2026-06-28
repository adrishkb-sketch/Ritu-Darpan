# scripts/downscale_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset

class DownscaleNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super(DownscaleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def load_config(config_path='pipeline-config.yaml'):
    if not os.path.exists(config_path):
        config_path = os.path.join('..', config_path)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_downscaler():
    cfg = load_config()
    processed_dir = cfg['paths']['processed_data_dir']
    output_file = cfg['paths']['output_file']
    
    # Standardize paths
    csv_path = os.path.join(processed_dir, output_file)
    if not os.path.exists(csv_path):
        if os.path.exists(os.path.join('..', csv_path)):
            csv_path = os.path.join('..', csv_path)
            processed_dir = os.path.join('..', processed_dir)
        else:
            print(f"Processed CSV not found at {csv_path}. Please run data-pipeline-main.py first.")
            return False

    print("Loading fused climate data for training downscaler model (memory-safe limit)...")
    df = pd.read_csv(csv_path, nrows=550000)
    
    if 'tmax' not in df.columns or 'lst' not in df.columns:
        print("Required columns (tmax, lst) for downscaling training are missing.")
        return False
        
    # Simulate a coarse temperature input by applying a spatial smoothing / adding noise
    # This represents the coarse 1.0 degree IMD temperature reading
    df['tmax_coarse'] = df['tmax'] + np.random.normal(0, 1.2, size=df.shape[0])
    
    # Feature columns: [tmax_coarse, lst, lat, lon]
    # Target column: [tmax]
    X_data = df[['tmax_coarse', 'lst', 'lat', 'lon']].values
    y_data = df[['tmax']].values
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Define dataset & dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    # Model, Loss, Optimizer
    model = DownscaleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Training DownscaleNet model in PyTorch...")
    model.train()
    epochs = 15
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss (MSE): {epoch_loss:.4f}")
        
    # Save the trained model
    models_dir = os.path.join(processed_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'downscaler.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Downscaler model weights saved to {model_path}")
    return True

def run_inference(coarse_temp, lst_temp, lats, lons):
    """
    Runs real-time inference to downscale temperature.
    Inputs can be scalars or numpy arrays.
    """
    cfg = load_config()
    processed_dir = cfg['paths']['processed_data_dir']
    if not os.path.exists(processed_dir):
        processed_dir = os.path.join('..', processed_dir)
        
    model_path = os.path.join(processed_dir, '..', 'models', 'downscaler.pth')
    
    model = DownscaleNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Flatten inputs if they are arrays
    c_flat = np.array(coarse_temp).flatten()
    lst_flat = np.array(lst_temp).flatten()
    lat_flat = np.array(lats).flatten()
    lon_flat = np.array(lons).flatten()
    
    # Create input feature matrix
    features = np.column_stack([c_flat, lst_flat, lat_flat, lon_flat])
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    with torch.no_grad():
        pred_flat = model(features_tensor).numpy().flatten()
        
    # Return matched shape
    if isinstance(coarse_temp, np.ndarray):
        return pred_flat.reshape(coarse_temp.shape)
    return pred_flat[0]

if __name__ == '__main__':
    train_downscaler()
