'''
This code population weighted annual mean of extreme heat days per IMAGE region.
Extreme heat days are defined here as the number of days above the historical (1995-2024)
threshold of 95th percentile of daily maximum temperature (T2M) calculated in the 
historical_95_percentile.py file.

The code receives a path to the historical 95 percentile threshold data, a path to the all the
CMIP6 model data, and a path to the population data and IMAGE regions.

NOTE: The code assumes the CMIP6 files are named frolling the format:
      tasmax_day_{model name}_{scenario}_{variant label}_{grid label}_{start date}-{end date}.nc
        where the start date and end date are in the format YYYYMMDD. This is the same format
        as when downloaded from the ESGF server.
'''

import utils

# Define path to historical 95 percentile threshold data
data_path = 'X:\\user\\liprandicn\\Data\\ERA5\\t2m_daily\\'
# Define path to CMIP6 model data 
model_path = 'X:\\user\\liprandicn\\Data\\CMIP6\\'
# Define path to population data and IMAGE regions
pop_path = 'X:\\user\\liprandicn\\Health Impacts Model\\Socioeconomic_Data\\Population\\GPOP\\'
# Define years for which to calculate the temperature index
years = [2025, 2030, 2050]

models_temperature_index = utils.temperature_index_all_models(model_path=model_path, 
                                                              data_path=data_path, 
                                                              pop_path=pop_path, 
                                                              years=years)
