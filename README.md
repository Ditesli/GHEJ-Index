# Extreme Heat Days Index
Code to obtain the number of future extreme heat days for SHAPE-RCP scenarios. Results used in Weber et al.

### historical_percentile.py
Uses historical ERA5 reanalysis daily temperature data (tmax, 1995-2024 period) to calculate the 90 and 95 percentile over a 30 year period. Returns two netcdf files with the percentile threshold per pixel.

### future_percentile.py
Uses ISIMIP climate projections and the netcdf files generated in $historical\_percentile.py$ to calculate the average number of days a region is exposed to extreme heat days. Calculations are done for a given year, RCP scenario and climate model. Final results are aggregated to [IMAGE regions](https://models.pbl.nl/image/Region_classification_map) via a population weighted average.

### utils.py
Contains all the relevant functions for calculations.


# Technical documentation

The Extreme heat days indicator is based on the Environmental Justice Index documentation by the U.S. Centers for Disease Control and Prevention
Agency for Toxic Substances and Disease Registry
Geospatial Research, Analysis, and Services Program (GRASP) (see documentation [here](https://www.atsdr.cdc.gov/place-health/php/eji/eji-technical-documentation.html#:~:text=Access%20the%20EJI%20Technical%20Documentation%20for%20detailed%20information,Technical%20Documentation.%20Download%20the%202022%20EJI%20Technical%20Documentation.)).

The extreme heat days are defined as the number of days above the 95th historical percentile. The historical threshold is taken here as the threshold of the 1995-2024 period. These values were obtained using maximum daily (2m) temperature from the [ERA5 reanalysis data](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=download), and calculating the 95th percentile per pixel (resolution of 0.25°x0.25°).

This historical thresholds were used to count the days that surpass such values using ScenarioMIP data (tasmax) for the scenarios SSP1-1.9, SSP1-2.6, SSP1-4.5 and the models: CAMS-CSM1-0, CNRM-ESM2-1, EC-Earth3, EC-Earth3-Veg, FGOALS-g3, GFDL-ESM4, GISS-E2-1-G, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0. Models that contain both scenarios and a varying resolution not lower than 250km. The index was calculated (per pixel) for the required years: 2025, 2030 and 2040.

The number of days per pixel was aggregated to every IMAGE region through the population weighted average. The code provides the model mean and the standard deviation.

