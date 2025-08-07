'''
This code calculates the historical 95th percentile of daily maximum temperature (T2M)
for the historical period  (defined here as 1995-2024) using ERA5 data. 
The code processes the data in latitude bands of 30 degrees to optimize memory usage.
'''


import utils
import numpy as np


# Set path to ERA5 data
data_path = ''
# Define the years range and step size for processing
years = np.arange(1995, 2025)
# Define the step size for latitude bands
step = 30

# Calculate historical percentiles
utils.calculate_historical_percentiles(data_path, years, step)





