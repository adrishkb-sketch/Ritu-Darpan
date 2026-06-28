# scripts/download_raw.py
import os
import yaml
import imdlib as imd

def load_config(config_path='pipeline-config.yaml'):
    if not os.path.exists(config_path):
        # Try parent directory if run from scripts folder
        config_path = os.path.join('..', config_path)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_imd_data(start_year, end_year, raw_imd_dir):
    """
    Downloads IMD rainfall, tmax, and tmin gridded binary datasets.
    """
    print(f"Starting IMD data download for years {start_year} to {end_year}...")
    
    # imdlib requires the subdirectories 'rain', 'tmax', 'tmin' to exist under the raw directory
    categories = {
        'rain': 'rain',
        'tmax': 'tmax',
        'tmin': 'tmin'
    }
    
    # Ensure raw directory exists
    os.makedirs(raw_imd_dir, exist_ok=True)
    
    for var_type in categories.keys():
        print(f"\n--- Downloading {var_type.upper()} data ---")
        try:
            # We set sub_dir=True to let imdlib create the subdirectory structure (rain, tmax, tmin) under raw_imd_dir
            # We set fn_format='yearwise' to save files as <year>.grd
            imd.get_data(
                var_type=var_type,
                start_yr=start_year,
                end_yr=end_year,
                fn_format='yearwise',
                file_dir=raw_imd_dir,
                sub_dir=True
            )
            print(f"[SUCCESS] Downloaded {var_type.upper()} data under {raw_imd_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to download {var_type.upper()} data: {e}")

if __name__ == "__main__":
    cfg = load_config()
    raw_imd = cfg['paths']['raw_imd_dir']
    
    # We default to a subset of years (e.g. 2018-2025) to prevent OOM errors in pandas during training, 
    # but the user can adjust these years as needed.
    start_yr = 2021
    end_yr = 2023
    
    download_imd_data(start_yr, end_yr, raw_imd)
