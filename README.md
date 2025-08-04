# temperature-extremes-index
Code to obtain future heat exposure under different SHAPE-RCP scenarios. Results used in Weber et al.

### historical_percentile.py
Uses historical ERA5 reanalysis daily temperature data (tmax, 1995-2024 period) to calculate the 90 and 95 percentile over a 30 year period. Returns two netcdf files with the percentile threshold per pixel.

### future_percentile.py
Uses ISIMIP climate projections and the netcdf files generated in $historical\_percentile.py$ to calculate the average number of days a region is exposed to extreme heat days. Calculations are done for a given year, RCP scenario and climate model. Final results are aggregated to [IMAGE regions](https://models.pbl.nl/image/Region_classification_map) via a population weighted average.

### utils.py
Contains all the relevant functions for calculations.