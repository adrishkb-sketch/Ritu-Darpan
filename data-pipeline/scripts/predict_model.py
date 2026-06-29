# scripts/predict_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset

class PredictNet(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64):
        super(PredictNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2) # Outputs: [predicted_rain, predicted_tmax]
        )
        
    def forward(self, x):
        return self.net(x)

def load_config(config_path='pipeline-config.yaml'):
    if not os.path.exists(config_path):
        config_path = os.path.join('..', config_path)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_predictor():
    cfg = load_config()
    processed_dir = cfg['paths']['processed_data_dir']
    output_file = cfg['paths']['output_file']
    
    csv_path = os.path.join(processed_dir, output_file)
    if not os.path.exists(csv_path):
        if os.path.exists(os.path.join('..', csv_path)):
            csv_path = os.path.join('..', csv_path)
            processed_dir = os.path.join('..', processed_dir)
        else:
            print(f"Processed CSV not found at {csv_path}. Please run data-pipeline-main.py first.")
            return False

    print("Loading fused climate data for training predictor model (memory-safe limit)...")
    df = pd.read_csv(csv_path, nrows=550000)
    
    required_cols = ['rain', 'tmax', 'tmin', 'rain_lag_1d', 'tmin_lag_1d', 'lat', 'lon']
    for col in required_cols:
        if col not in df.columns:
            # Generate lag column if missing
            df = df.sort_values(by=['lat', 'lon', 'datetime'])
            grouped = df.groupby(['lat', 'lon'])
            if 'rain' in df.columns and col == 'rain_lag_1d':
                df['rain_lag_1d'] = grouped['rain'].shift(1).fillna(0)
            if 'tmin' in df.columns and col == 'tmin_lag_1d':
                df['tmin_lag_1d'] = grouped['tmin'].shift(1).fillna(0)
                
    # Sort and shift for target values (predicting next day's values)
    df = df.sort_values(by=['lat', 'lon', 'datetime'])
    grouped = df.groupby(['lat', 'lon'])
    
    # Target: Tomorrow's rain and tmax
    df['target_rain'] = grouped['rain'].shift(-1)
    df['target_tmax'] = grouped['tmax'].shift(-1)
    
    # Drop rows with NaNs (which will be the last day of each coordinate)
    df_clean = df.dropna(subset=['target_rain', 'target_tmax'])
    
    # Feature columns: [rain_lag_1d, tmin_lag_1d, tmax, rain, lat, lon, elevation, tmax_grad_x, tmax_grad_y, rain_grad_x, rain_grad_y, tmax_spatial_mean, rain_spatial_mean]
    X_cols = ['rain_lag_1d', 'tmin_lag_1d', 'tmax', 'rain', 'lat', 'lon', 'elevation', 'tmax_grad_x', 'tmax_grad_y', 'rain_grad_x', 'rain_grad_y', 'tmax_spatial_mean', 'rain_spatial_mean']
    X_data = df_clean[X_cols].values
    y_data = df_clean[['target_rain', 'target_tmax']].values
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Define dataset & dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    # Model, Loss, Optimizer
    model = PredictNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Training PredictNet model in PyTorch...")
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
    model_path = os.path.join(models_dir, 'predictor.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Predictor model weights saved to {model_path}")
    return True

def run_forecast(current_features, mc_dropout=False):
    """
    Inference function with optional Monte Carlo Dropout.
    current_features: numpy array shape (N, 13) containing [rain_lag, tmin_lag, tmax, rain, lat, lon, elevation, grad_x, grad_y, etc.]
    Returns shape (N, 2) containing forecasted [rain, tmax] for the next day.
    """
    cfg = load_config()
    processed_dir = cfg['paths']['processed_data_dir']
    if not os.path.exists(processed_dir):
        processed_dir = os.path.join('..', processed_dir)
        
    model_path = os.path.join(processed_dir, '..', 'models', 'predictor.pth')
    
    model = PredictNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
    if mc_dropout:
        model.train() # Enable dropout at test time
    else:
        model.eval()
    
    features_tensor = torch.tensor(current_features, dtype=torch.float32)
    with torch.no_grad():
        forecast = model(features_tensor).numpy()
    
    # Clip negative forecasted rain values to 0
    forecast[:, 0] = np.clip(forecast[:, 0], 0, None)
    return forecast

if __name__ == '__main__':
    train_predictor()
