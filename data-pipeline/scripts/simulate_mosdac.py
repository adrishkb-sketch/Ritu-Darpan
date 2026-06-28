# scripts/simulate_mosdac.py
import os
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import datetime
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

def load_config(config_path='pipeline-config.yaml'):
    if not os.path.exists(config_path):
        config_path = os.path.join('..', config_path)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_days_in_range(start_year, end_year):
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    delta = end_date - start_date
    return [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]

def generate_satellite_data():
    cfg = load_config()
    spatial = cfg['spatial_settings']
    lat_min, lat_max = spatial['lat_min'], spatial['lat_max']
    lon_min, lon_max = spatial['lon_min'], spatial['lon_max']
    
    temporal = cfg['temporal_settings']
    start_year = temporal['start_year']
    end_year = temporal['end_year']
    
    raw_imd_dir = cfg['paths']['raw_imd_dir']
    raw_mosdac_dir = cfg['paths']['raw_mosdac_dir']
    
    # Standardize path references
    if not os.path.exists(raw_mosdac_dir):
        # relative path resolution if running from script folder
        if os.path.exists(os.path.join('..', raw_mosdac_dir)):
            raw_mosdac_dir = os.path.join('..', raw_mosdac_dir)
            raw_imd_dir = os.path.join('..', raw_imd_dir)
        else:
            os.makedirs(raw_mosdac_dir, exist_ok=True)
            
    os.makedirs(raw_mosdac_dir, exist_ok=True)
    
    days = get_days_in_range(start_year, end_year)
    print(f"Loading IMD ground data to seed satellite simulation for {len(days)} days...")
    
    # Load IMD data into memory for interpolation
    # Since we cropped to West Bengal, they fit in memory easily
    imd_datasets = {}
    for var in ['rain', 'tmax', 'tmin']:
        var_dfs = []
        var_dir = os.path.join(raw_imd_dir, var)
        if os.path.exists(var_dir):
            for year in range(start_year, end_year + 1):
                grd_file = os.path.join(var_dir, f"{year}.grd")
                if not os.path.exists(grd_file):
                    grd_file = os.path.join(var_dir, f"{year}.GRD")
                if os.path.exists(grd_file):
                    try:
                        import imdlib as imd
                        data = imd.open_data(var, year, year, 'yearwise', raw_imd_dir)
                        # Crop to West Bengal bounds
                        ds = data.get_xarray().sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                        var_dfs.append(ds)
                    except Exception as e:
                        print(f"Error reading IMD {var} for {year}: {e}")
        if var_dfs:
            # Combine across years
            imd_datasets[var] = xr.concat(var_dfs, dim='time')
            print(f"Successfully loaded IMD {var} data. Time steps: {len(imd_datasets[var].time)}")
            
    # Establish target satellite dense grid (0.04 degree resolution, ~4km)
    sat_lats = np.arange(lat_min, lat_max, 0.04)
    sat_lons = np.arange(lon_min, lon_max, 0.04)
    grid_lon, grid_lat = np.meshgrid(sat_lons, sat_lats)
    
    # Simple Mock Topography/Elevation for West Bengal (high in North - Himalayas, low in South - Sundarbans)
    # North latitude is ~27.5, South is ~21.0
    elevation_grid = (grid_lat - 21.0) / (27.5 - 21.0) # normalized 0 to 1
    elevation_grid = np.where(grid_lat > 26.5, elevation_grid * 3000, elevation_grid * 150) # mountainous North
    
    print("\nStarting HDF5 Satellite Data Simulation...")
    for day in tqdm(days):
        day_str = day.strftime("%Y%m%d")
        dt_stamp = pd.to_datetime(day)
        
        # 1. Fetch seed data for LST (temp) and Rainfall from IMD
        # Default climatology values if IMD file doesn't cover this day
        base_tmax = 30.0
        base_tmin = 20.0
        base_rain = 0.0
        
        # Check if we have loaded IMD data for this timestamp
        has_tmax = 'tmax' in imd_datasets and dt_stamp in imd_datasets['tmax'].time
        has_tmin = 'tmin' in imd_datasets and dt_stamp in imd_datasets['tmin'].time
        has_rain = 'rain' in imd_datasets and dt_stamp in imd_datasets['rain'].time
        
        # Generate spatial seed arrays by interpolating coarse IMD data
        if has_tmax:
            slice_tmax = imd_datasets['tmax'].sel(time=dt_stamp)['tmax'].values
            # Create interpolator
            interp = RegularGridInterpolator(
                (imd_datasets['tmax'].lat.values, imd_datasets['tmax'].lon.values),
                slice_tmax, method='linear', bounds_error=False, fill_value=None
            )
            seed_tmax = interp((grid_lat, grid_lon))
        else:
            seed_tmax = np.full(grid_lat.shape, base_tmax)
            
        if has_tmin:
            slice_tmin = imd_datasets['tmin'].sel(time=dt_stamp)['tmin'].values
            interp = RegularGridInterpolator(
                (imd_datasets['tmin'].lat.values, imd_datasets['tmin'].lon.values),
                slice_tmin, method='linear', bounds_error=False, fill_value=None
            )
            seed_tmin = interp((grid_lat, grid_lon))
        else:
            seed_tmin = np.full(grid_lat.shape, base_tmin)
            
        if has_rain:
            slice_rain = imd_datasets['rain'].sel(time=dt_stamp)['rain'].values
            interp = RegularGridInterpolator(
                (imd_datasets['rain'].lat.values, imd_datasets['rain'].lon.values),
                slice_rain, method='linear', bounds_error=False, fill_value=0.0
            )
            seed_rain = interp((grid_lat, grid_lon))
        else:
            seed_rain = np.full(grid_lat.shape, base_rain)
            
        # 2. Simulate INSAT Land Surface Temperature (LST)
        # LST = Air Temp + Solar Radiation effect - Elevation effect + High-Resolution Noise
        lst_data = (seed_tmax + seed_tmin) / 2.0
        lst_data = lst_data - (elevation_grid * 0.0065) # standard lapse rate (-6.5C per 1km)
        # Add high-res spatial noise (simulating sub-grid microclimates)
        noise_lst = np.random.normal(0, 1.5, size=grid_lat.shape)
        lst_data = lst_data + noise_lst + 5.0 # LST is typically higher than air temp during day
        # Apply standard mask (e.g. keep values reasonable)
        lst_data = np.clip(lst_data, 5, 55)
        
        # 3. Simulate INSAT Sea Surface Temperature (SST)
        # Bay of Bengal is in the South, roughly below latitude 22.0°N
        sst_data = np.full(grid_lat.shape, np.nan)
        sea_mask = grid_lat < 22.0
        sst_data[sea_mask] = 28.5 + np.random.normal(0, 0.5, size=np.sum(sea_mask))
        # Mask out LST in sea regions
        lst_data[sea_mask] = np.nan
        
        # 4. Simulate INSAT Satellite Rainfall (IMC)
        # Satellite rain matches IMD rain but with wider, smoother cloud coverage + radar noise
        imc_data = seed_rain * (1.0 + np.random.normal(0, 0.2, size=grid_lat.shape))
        imc_data = np.clip(imc_data, 0, None)
        # Smooth spatial representation
        from scipy.ndimage import gaussian_filter
        imc_data = gaussian_filter(imc_data, sigma=1.0)
        
        # Save products as separate HDF5 files to match MOSDAC naming conventions
        # Format: 3RIMG_YYYYMMDD_L2B_LST.h5 etc
        acq_time_str = f"{day.strftime('%Y-%m-%d')}T12:00:00Z"
        
        # Helper to write standard MOSDAC HDF5
        def write_h5_file(prod_key, data_array, filename):
            file_path = os.path.join(raw_mosdac_dir, filename)
            # Remove existing if any
            if os.path.exists(file_path):
                os.remove(file_path)
            with h5py.File(file_path, 'w') as f:
                f.attrs['Acquisition_Time'] = acq_time_str.encode('utf-8')
                f.create_dataset('Latitude', data=grid_lat)
                f.create_dataset('Longitude', data=grid_lon)
                geo_group = f.create_group('Geophysical_Data')
                geo_group.create_dataset(prod_key, data=data_array)
                
        write_h5_file('LST', lst_data, f"3RIMG_{day_str}_L2B_LST.h5")
        write_h5_file('SST', sst_data, f"3RIMG_{day_str}_L2B_SST.h5")
        write_h5_file('Rainfall', imc_data, f"3RIMG_{day_str}_L2B_IMC.h5")
        
    print(f"Successfully generated 3 years of daily HDF5 satellite files inside: {raw_mosdac_dir}")

if __name__ == '__main__':
    generate_satellite_data()
