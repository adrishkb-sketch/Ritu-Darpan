# build_data.py
import os
import sys
import subprocess

def run_command(args, cwd=None):
    """Runs a system command and streams output."""
    print(f"\nRunning: {' '.join(args)} (in {cwd or '.'})...")
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output to terminal
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"\n[ERROR] Command failed with return code {return_code}")
        sys.exit(return_code)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.join(root_dir, 'data-pipeline')
    
    print("=" * 60)
    print(" Ritu-Darpan Data Pipeline Setup & Build Script")
    print("=" * 60)
    
    # 1. Verify dependencies
    print("\n[Step 1/5] Verifying dependencies...")
    try:
        import pandas
        import numpy
        import xarray
        import h5py
        import imdlib
        import yaml
        import scipy
        import shapely
        import tqdm
        import pyarrow
        import streamlit
        import torch
        import plotly
        print(" -> All required packages are installed!")
    except ImportError as e:
        print(f" -> Missing dependency: {e.name}")
        print("\nPlease run the following command to install all requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
        
    # 2. Change working directory to 'data-pipeline'
    if not os.path.exists(pipeline_dir):
        print(f"[ERROR] Could not find data-pipeline folder at {pipeline_dir}")
        sys.exit(1)
        
    os.chdir(pipeline_dir)
    print(f" -> Changed working directory to {os.getcwd()}")
    
    # 3. Download Raw IMD Data
    print("\n[Step 2/5] Downloading raw IMD climate datasets...")
    run_command([sys.executable, os.path.join('scripts', 'download_raw.py')])
    
    # 4. Simulate MOSDAC Satellite Data
    print("\n[Step 3/5] Simulating MOSDAC satellite data...")
    run_command([sys.executable, os.path.join('scripts', 'simulate_mosdac.py')])
    
    # 5. Run the Data Pipeline Orchestrator
    print("\n[Step 4/5] Running main data-pipeline processing...")
    run_command([sys.executable, 'data-pipeline-main.py'])
    
    # 6. Convert Output to Parquet
    print("\n[Step 5/5] Converting CSV dataset to Parquet for dashboard performance...")
    run_command([sys.executable, os.path.join('scratch', 'csv_to_parquet.py')])
    
    print("\n" + "=" * 60)
    print("🎉 Success! The data pipeline has been successfully built!")
    print("=" * 60)
    print("\nYou can now launch the dashboard by running:")
    print("  streamlit run dashboard/app.py")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
