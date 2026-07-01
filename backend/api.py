import os
import sys
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# Add data-pipeline to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data-pipeline')))
try:
    from scripts.downscale_model import DownscaleNet
    from scripts.predict_model import PredictNet
except ImportError:
    print("Warning: Could not import model classes. Ensure data-pipeline is built.")
    DownscaleNet, PredictNet = None, None

app = FastAPI(title="Ritu-Darpan Climate Inference API", description="Microservice for PyTorch Climate Models")

# Load Models
downscale_model = None
predict_model = None

@app.on_event("startup")
def load_models():
    global downscale_model, predict_model
    if DownscaleNet is None or PredictNet is None:
        return
        
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'data-pipeline', 'models')
    downscale_path = os.path.join(models_dir, 'downscaler.pth')
    predict_path = os.path.join(models_dir, 'predictor.pth')

    downscale_model = DownscaleNet()
    if os.path.exists(downscale_path):
        downscale_model.load_state_dict(torch.load(downscale_path, map_location=torch.device('cpu')))
    downscale_model.eval()

    predict_model = PredictNet()
    if os.path.exists(predict_path):
        predict_model.load_state_dict(torch.load(predict_path, map_location=torch.device('cpu')))
    predict_model.eval()
    print("PyTorch models loaded successfully into FastAPI.")


@app.get("/")
def read_root():
    return {
        "status": "online", 
        "message": "Ritu-Darpan API is running perfectly! The dashboard will communicate with this service automatically in the background.",
        "documentation": "To test the API endpoints directly, go to http://127.0.0.1:8000/docs"
    }


class SimulationRequest(BaseModel):
    downscale_features: List[List[float]] # Shape: (N, 5)
    predict_features: List[List[float]] # Shape: (N, 13)
    num_mc_runs: int = 15

@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    if downscale_model is None or predict_model is None:
        return {"error": "Models not loaded. Check backend configuration."}
        
    X_downscale = torch.tensor(req.downscale_features, dtype=torch.float32)
    X_predict = torch.tensor(req.predict_features, dtype=torch.float32)
    
    # 1. Run Downscaler
    downscale_model.eval()
    with torch.no_grad():
        downscaled_tmax = downscale_model(X_downscale).numpy().flatten().tolist()
        
    # 2. Run Predictor (MC Dropout Ensemble)
    ensemble_preds = []
    predict_model.train() # Enable dropout layers for test-time UQ
    for _ in range(req.num_mc_runs):
        with torch.no_grad():
            preds = predict_model(X_predict).numpy()
            ensemble_preds.append(preds.tolist())
            
    return {
        "downscaled_tmax": downscaled_tmax,
        "ensemble_preds": ensemble_preds
    }
