import numpy as np
import pandas as pd
import xarray as xr
import re
import glob
import os



def calculate_historical_percentiles(data_path: str, years: np.ndarray, step: int):
    
    """
    Calculate the 95th percentiles of daily maximum temperature from ERA5 
    reanalysis data.
    File resolution is 0.25 degrees (720x1440 dimension).
    Files were renamed following the format: 'era5_t2m_max_day_<year>.nc'.

    Parameters:
    - data_path: str, path to the directory containing the ERA5 data files.
    - years: np.ndarray, array of years for which to calculate percentiles, 
      usually a 30 year period.
    - step: int, step size for latitude bands to optimize memory usage.
      The data is processed in bands of latitude to avoid memory allocation issues.
      Depending on the available memory, a step of 20 is recommended. 
      If unable to allocate memory to process, decrease the step size.

    Returns:
    - None, saves the results as NetCDF files.
    """
    
    # Process data in latitude bands to optimize memory usage
    lats = range(0, 720, step)  

    p95_bands = [] #; p90_bands = []

    # Iterate over each latitude band
    for lat in lats:
        
        print(f"Processing latitudes: {lat} - {lat + step}")
        
        temporal_lat_data = []
        
        for year in years:
            
            # Load the dataset for the specific year and latitude
            data_name = f'era5_t2m_max_day_{year}.nc'
            with xr.open_dataset(data_path + data_name) as data:
                data = data['t2m'].isel(latitude=slice(lat, lat + step))
                temporal_lat_data.append(data.load())
            
        # Concatenate the data for the current latitudes across all years
        temporal_data = xr.concat(temporal_lat_data, dim='valid_time')
        
        # Calculate the 95th percentile for the latitude band and append it to the list
        p95_band = temporal_data.quantile(0.95, dim='valid_time')
        p95_bands.append(p95_band)
        
        # p90_band = temporal_data.quantile(0.90, dim='valid_time')
        # p90_bands.append(p90_band)
        
    p95_final = xr.concat(p95_bands, dim='latitude')
    p95_final.name = 't2m_max_p95'
    p95_final.to_netcdf(data_path + f'era5_t2m_max_{years[0]}-{years[-1]}_p95.nc')
    
    # p90_final = xr.concat(p90_bands, dim='latitude')
    # p90_final.name = 't2m_max_p90'
    # p90_final.to_netcdf(data_path + f'era5_t2m_max_{years[0]}-{years[-1]}_p90.nc')



def align_data(data: xr.Dataset, celsius=False, longitude_shift=False, standar_names=False):
    
    """
    Aligns the data by converting temperature to Celsius and shifting longitudes if needed.
    
    Parameters:
    - data: xr.Dataset, the dataset to align.
    - celsius: bool, if True, converts temperature from Kelvin to Celsius.
    - longitude_shif: bool, if True, shifts longitudes from 0-360 to -180-180.
    
    Returns:
    - xr.Dataset, the aligned dataset.
    """
    
    if celsius:
        data = data - 273.15  # Convert from Kelvin to Celsius
        
    if standar_names:
        data = homogenize_lat_lon(data)
        
    if longitude_shift:
        data = data.assign_coords(longitude=((data.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")

    return data



def homogenize_lat_lon(ds: xr.Dataset, new_lat='latitude', new_lon='longitude', new_time='valid_time'):
    
    """
    Renames latitude and longitude coordinates to standard ERA5 names.

    Parameters:
    - ds: xarray.Dataset
    - lat_names: list of possible latitude coordinate names
    - lon_names: list of possible longitude coordinate names
    - new_lat: desired standard name for latitude
    - new_lon: desired standard name for longitude

    Returns:
    - xarray.Dataset with standardized coordinate names
    """
    
    lat_names = ['lat', 'Lat', 'LAT', 'nav_lat', 'lat_bnds']
    lon_names = ['lon', 'Lon', 'LON', 'nav_lon', 'lon_bnds']
    time_names = ['time', 'Time', 'time_bnds']

    rename_dict = {}
    for name in ds.coords:
        if name in lat_names:
            rename_dict[name] = new_lat
        elif name in lon_names:
            rename_dict[name] = new_lon
        elif name in time_names:
            rename_dict[name] = new_time

    return ds.rename(rename_dict)



def pop_and_regions(file_path: str, scenario: str, year: int, tas_max: xr.Dataset):
    
    '''Read scenario-dependent population data, interpolate it to yearly data, select specific year
    reduce resolution if coarse is True.
    Read netcdf file with region classification map.
    
    Parameters:
    - wdir: str, working directory where the population data is stored.
    - scenario: str, the socioeconomic scenario for which to load population data.
    - coarse: bool, if True, reduces the resolution of the population data.
    - start_year: int, the starting year for the population data.
    - end_year: int, the ending year for the population data.
    
    Returns:
    - xr.Dataset, the population data interpolated to yearly resolution.
    - xr.Dataset, the region classification map.
    '''
    
    # Load the population data from the specified file path
    pop = xr.open_dataset(file_path+'GPOP_SSP1_M.nc')
    # Select yearly data for the specified scenario
    pop_year = pop.sel(time=f'{year}', method='nearest')
    # Interpolate lat and lon to match the model grid
    pop_year = pop_year.interp(longitude=tas_max.longitude, latitude=tas_max.latitude)
    
    # Load the region classification map
    greg = xr.open_dataset(file_path+'GREG.nc')
    # Interpolate the region map to match the model grid
    greg = greg.interp(longitude=tas_max.longitude, latitude=tas_max.latitude, method='nearest')
    # Average over time and height dimensions (not needed for this analysis)
    greg = greg.mean(dim='time')
    
    return pop_year, greg



def get_region_values(excedance_count: xr.Dataset, pop_year: xr.Dataset, 
                      region_mask: xr.Dataset, model_file: str, year: int):
    
    """
    Calculate the total number of days exceeding the 95th percentile and the population exposure 
    for a specific region.

    Parameters:
    - exceedance_count: xr.Dataset, dataset containing the count of days exceeding the 95th percentile.
    - pop_year: xr.Dataset, dataset containing the population data for the specified year.
    - region_mask: xr.Dataset, dataset containing the mask for the specific region.

    Returns:
    - tuple of (total_exceedance_days, total_population_exposure)
    """
    
    # Extract the model name from the file name using regex
    model_name = re.search(r'day_(.*)_r', model_file).group(1) 
    
    # Flatten the datasets to create a DataFrame for easier manipulation
    region_array = region_mask.GREG.values.flatten()
    excedance_array = excedance_count.t2m_max_p95.values.flatten()
    pop_array = pop_year.GPOP.values.flatten()
    
    # Create a DataFrame with the flattened data
    regions_df = pd.DataFrame({
        'IMAGE_region': region_array,
        'T95': excedance_array,
        'population': pop_array
    })
    
    # Remove rows where the region is NaN
    regions_df = regions_df[~regions_df['IMAGE_region'].isna()]
    
    # Group by region and calculate the weighted average of the 95th percentile exceedance
    regions_df = (
        regions_df
        .groupby('IMAGE_region')
        .apply(lambda x: (x['T95'] * x['population']).sum() / x['population'].sum())
        .reset_index(name=f'{year}_{model_name}')
        )

    return regions_df



def temperature_index(years, model_path, data_path, pop_file, final_data, model_file_name):
    
    '''
    Calculate the number of days exceeding the 95th percentile of daily maximum temperature
    for a given scenario and year using a specific model's data.
    
    Parameters:
    - scenario: str, the socioeconomic scenario (e.g., 'ssp119', 'ssp126').
    - year: int, the year for which to calculate the index.
    - model_path: str, path to the directory containing the model data files.
    - model_file_name: str, the name of the model file to be processed.
    - data_path: str, path to the directory containing the historical data files.
    - pop_file: str, path to the directory containing the population data files.
    - p95_hist: xr.Dataset, dataset containing the historical 95th percentile data.
    
    Returns:
    - final_regions: pd.DataFrame, DataFrame containing the total number of days exceeding the 95th percentile
    and the population exposure for each region.
    '''
    
    # p90_hist = xr.open_dataset(data_path + 'p90.nc')
    p95_hist = xr.open_dataset(data_path + 'p95.nc')
    
    # Load the dataset for the specific model, scenario, and year
    tas_max = xr.open_dataset(model_path + model_file_name)
    
    # Align the data to ensure consistency in units and coordinates
    tas_max = align_data(tas_max['tasmax'], celsius=True, longitude_shift=True, standar_names=True)
    
    # Extract the start and end years from the model file name using regex
    match = re.search(r'_(\d{8})-(\d{8})', model_file_name)
    start_year = int(match.group(1)[:4])
    end_year = int(match.group(2)[:4])
    
    # Create a range of years from the start to end year
    year_range = np.arange(start_year, end_year + 1)
    
    # Iterate over the specified years and calculate the index
    for year in years:
        # Check if the year is within the range of years in the model file
        if year in year_range:
    
            # Select the tasmax variable and filter by year
            tas_max_year = tas_max.sel(valid_time=slice(f'{year}-01-01', f'{year}-12-31'))
            
            # Interpolate the p95 historical data to match the tas_max grid        
            p95_hist = p95_hist.interp(longitude=tas_max.longitude, latitude=tas_max.latitude)
            
            # Calculate the number of days that exceed the 95th percentile
            excedance_count = (tas_max_year > p95_hist).sum(dim='valid_time')
            
            # Calculate the population exposure
            population, image_regions = pop_and_regions(pop_file, 'SSP1', year, tas_max_year)
            
            # Get the total number of days exceeding the 95th percentile and the population exposure for each region
            final_regions = get_region_values(excedance_count, population, image_regions, model_file_name, year)

            # Merge the results into the final DataFrame
            final_data = final_data.merge(final_regions, on='IMAGE_region', how='outer')
            
    return final_data, year



def temperature_index_all_models(model_path, data_path, pop_path, years):
    
    # Initialize an empty DataFrame to hold the temperature index data for each model
    models_data = pd.DataFrame(index=pd.Index(np.arange(1,28,1.), name='IMAGE_region'))

    # Get all the netCDF files for tasmax in the specified model path
    files = glob.glob(os.path.join(model_path, 'tasmax_day_*.nc'))

    # Loop through each file and extract the temperature index data for the specified years
    for file in files:
        models_data, year = temperature_index(years, model_path, data_path, pop_path, models_data, os.path.basename(file))
        print(f'Processed {file} for year {year}')
        

    # Extract the year, model, and scenario from the column names
    column_info = models_data.columns.str.extract(r'(?P<year>\d{4})_(?P<model>.+?)_(?P<scenario>.+)')

    # Create a DataFrame to hold the mean and standard deviation of each model
    df_mean = pd.DataFrame(index=models_data.index)
    df_std = pd.DataFrame(index=models_data.index)

    # Create a new column for the model names in the DataFrame
    group_keys = column_info['year'].astype(str) + '_' + column_info['scenario'].astype(str)

    # Calculate the mean and standard deviation for each group of models
    for group in group_keys.unique():
        cols_in_group = models_data.columns[group_keys == group]
        df_mean[group + '_model-mean'] = models_data[cols_in_group].mean(axis=1)
        df_std[group + '_model-std'] = models_data[cols_in_group].std(axis=1)

    # Combine the mean and standard deviation DataFrames
    df_summary = pd.concat([df_mean, df_std], axis=1)
    df_summary = df_summary.set_index('nan_nan_model-mean')
    df_summary.index = df_summary.index.astype(int)

    # Load the IMAGE regions names
    image_names = pd.read_csv(pop_path + 'IMAGE_regions.csv', index_col=0)

    # Merge the region names with the summary DataFrame
    temperature_index_models = pd.merge(image_names['Region'], df_summary, left_index=True, right_index=True)
    
    return temperature_index_models