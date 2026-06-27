
import pandas as pd
import numpy as np
import xarray as xr
import h5py
import imdlib as imd
import yaml
import os

def load_config(config_path='../pipeline_config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def process_imd_binary(year, var_type, raw_dir):
    """
    Reads IMD binary gridded files using imdlib.
    var_type must be one of: 'rain', 'tmax', 'tmin'
    """
    # Check if the expected binary file exists
    # imdlib expects files to be named <year>.grd for rainfall or <year>.bin for temp sometimes
    # but more accurately it looks in the raw_dir.
    # We'll do a simple check to see if there are any files in the directory
    if not os.listdir(raw_dir):
        print(f"No files found in {raw_dir}. Skipping IMD {var_type} for {year}.")
        return pd.DataFrame()

    data = imd.open_data(var_type, year, year, 'yearwise', raw_dir)
    
    # Convert xarray dataset to a flattened pandas DataFrame
    df = data.get_xarray().to_dataframe().reset_index()
    
    # Standardize column names
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['source'] = 'imd_ground'
    
    # Drop masked/invalid values (IMD uses -999.0 for no-data regions like oceans)
    df = df[df[var_type] > -100.0]
    
    return df

def process_mosdac_h5(file_path, product_key):
    """
    Reads MOSDAC INSAT-3D/3DR HDF5 files.
    product_key defines which layer to extract (e.g., 'LST', 'SST', 'IMC')
    """
    print(f"Ingesting MOSDAC {product_key} from {os.path.basename(file_path)}...")
    
    with h5py.File(file_path, 'r') as h5_data:
        # MOSDAC HDF5 structure: Latitude and Longitude are usually 2D arrays 
        # stored under the 'Geolocation' or root group. Data is under 'Geophysical_Data'
        
        try:
            lats = np.array(h5_data['Latitude'])
            lons = np.array(h5_data['Longitude'])
            
            # The exact path depends on the product (3RIMGL2BLST vs 3RIMGL2BIMC)
            if product_key == 'LST':
                layer = np.array(h5_data['Geophysical_Data']['LST'])
            elif product_key == 'SST':
                layer = np.array(h5_data['Geophysical_Data']['SST'])
            elif product_key == 'IMC':
                layer = np.array(h5_data['Geophysical_Data']['Rainfall'])
            else:
                raise ValueError("Unknown MOSDAC product key")
                
            # Extract timestamp from HDF5 attributes (usually stored as 'Acquisition_Time')
            # Fallback to dummy time for demonstration if attribute is missing
            acq_time = h5_data.attrs.get('Acquisition_Time', b'2026-06-27T00:00:00Z').decode('utf-8')
            
        except KeyError as e:
            print(f"Error reading HDF5 structure: {e}")
            return pd.DataFrame()

    # Flatten the 2D arrays into a 1D list of records
    df = pd.DataFrame({
        'lat': lats.flatten(),
        'lon': lons.flatten(),
        product_key.lower(): layer.flatten(),
        'datetime': pd.to_datetime(acq_time)
    })
    
    df['source'] = 'insat_satellite'
    
    # Filter out nodata/fill values (MOSDAC often uses 32767 or NaN)
    df = df.dropna()
    df = df[df[product_key.lower()] < 1000] 
    
    return df

def build_master_grid(config):
    """Orchestrates the ingestion of all ground and satellite datasets."""
    master_df_list = []
    
    # 1. Process IMD Ground Data (Example: Year 2026)
    imd_dir = config['paths']['raw_imd_dir']
    if os.path.exists(imd_dir):
        # Requires files to be pre-downloaded or fetched via imdlib.get_data()
        df_rain = process_imd_binary(2026, 'rain', imd_dir)
        df_tmax = process_imd_binary(2026, 'tmax', imd_dir)
        master_df_list.extend([df_rain, df_tmax])

    # 2. Process MOSDAC Satellite Data
    mosdac_dir = config['paths']['raw_mosdac_dir']
    if os.path.exists(mosdac_dir):
        for file in os.listdir(mosdac_dir):
            if file.endswith('.h5'):
                file_path = os.path.join(mosdac_dir, file)
                
                # Determine product type from file name (e.g., 3RIMGL2BLST -> LST)
                if 'LST' in file:
                    df = process_mosdac_h5(file_path, 'LST')
                elif 'SST' in file:
                    df = process_mosdac_h5(file_path, 'SST')
                elif 'IMC' in file:
                    df = process_mosdac_h5(file_path, 'IMC')
                
                master_df_list.append(df)

    if not master_df_list:
        print("No data processed. Check your raw data directories.")
        return pd.DataFrame()

    # 3. Merge all records
    print("Consolidating datasets...")
    final_df = pd.concat(master_df_list, ignore_index=True)
    return final_df

if __name__ == "__main__":
    cfg = load_config()
    master_grid = build_master_grid(cfg)
    
    if not master_grid.empty:
        output_path = os.path.join(cfg['paths']['processed_data_dir'], cfg['paths']['output_file'])
        master_grid.to_csv(output_path, index=False)
        print(f"Data ingestion complete. Output saved to {output_path}")