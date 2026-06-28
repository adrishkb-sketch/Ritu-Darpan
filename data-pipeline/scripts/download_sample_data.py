# scripts/download_sample_data.py
import imdlib as imd
import os
import yaml
from ingest_raw import load_config

def download_imd_data(config):
    """
    Downloads sample IMD data for the specified years.
    """
    print("Initializing IMD data download...")
    
    # Get paths from config
    raw_imd_dir = config['paths']['raw_imd_dir']
    
    # imdlib downloads into variable subdirectories if sub_dir=True
    # We want to keep it simple for the user
    years = [2024, 2025]
    variables = ['rain', 'tmax']
    
    for var in variables:
        print(f"Downloading {var} data for years {years}...")
        try:
            # imdlib.get_data handles the downloading
            # fn_format='yearwise' saves files as <year>.grd or <year>.bin
            data = imd.get_data(var, years[0], years[-1], fn_format='yearwise', file_dir=raw_imd_dir)
            print(f"[OK] Successfully downloaded {var} data.")
        except Exception as e:
            print(f"[ERROR] Failed to download {var} data: {e}")

if __name__ == "__main__":
    # Load config relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, '..', 'pipeline-config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Ensure the directory exists
    target_dir = config['paths']['raw_imd_dir']
    os.makedirs(target_dir, exist_ok=True)
    
    download_imd_data(config)
