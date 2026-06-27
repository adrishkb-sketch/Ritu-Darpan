
import sys
import os

print("--- Verifying Imports ---")
try:
    import pandas as pd
    import numpy as np
    import yaml
    from sqlalchemy import create_engine
    from scipy.interpolate import griddata
    print("[OK] Core dependencies found.")
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")

try:
    from scripts.ingest_raw import load_config, build_master_grid
    from scripts.clean_anomalies import clean_pipeline
    from scripts.feature_eng import run_feature_engineering
    print("[OK] Local modules found and importable.")
except Exception as e:
    print(f"[ERROR] Module import error: {e}")

print("\n--- Verifying Config ---")
if os.path.exists('pipeline-config.yaml'):
    try:
        config = load_config('pipeline-config.yaml')
        print("[OK] Config loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Config error: {e}")
else:
    print("[ERROR] Config file missing.")

print("\n--- Verifying Main Orchestrator ---")
try:
    import subprocess
    result = subprocess.run([sys.executable, 'data-pipeline-main.py'], capture_output=True, text=True)
    if "Workflow stopped: No data ingested." in result.stdout or "Pipeline run successful!" in result.stdout:
         print("[OK] Main orchestrator ran successfully (graceful exit on no data).")
    else:
        print(f"[ERROR] Main orchestrator failed with output:\n{result.stdout}\n{result.stderr}")
except Exception as e:
    print(f"[ERROR] Error running orchestrator: {e}")
