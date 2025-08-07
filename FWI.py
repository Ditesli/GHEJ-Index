'''
The code calculates the Fire Weather Index (FWI) for different CMIP6 models and scenarios in each 
IMAGE region. The FWI is a measure of fire danger based on meteorological conditions, based on the the paper:
https://doi.org/10.5194/essd-15-2153-2023
FWI data projections is provided by the authors of the paper, and is available at:
https://doi.org/10.3929/ethz-b-000583391
'''


import xarray as xr
import utils


# Define path to CMIP6 model data 
model_path = ''
# Define path to population data and IMAGE regions
pop_path = ''
# Define years for which to calculate the temperature index
years = [2025, 2030, 2050]
# Define the population scenario for which to calculate the temperature index
scenario = 'SSP2_CP' # SSP1_M, SSP2_CP, SSP3_H, SSP5_H	


utils.index_all_models(index_type='fwi',
                       model_path=model_path, 
                       pop_path=pop_path, 
                       years=years,
                       scenario=scenario, 
                       data_path=None) 