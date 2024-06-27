# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob as glob
import os

# %%
# set the filepath
filepath = '/storage/dlhogan/synoptic_sublimation/splash_data/'
if not os.path.exists(filepath+'ceilometer'):
    print('Data not downlaoded yet. Would you like to download it?')
    download = input('y/n: ')
    if download == 'y':
        # downlaod the data from https://zenodo.org/records/10520198/files/ckp.cl51.cloud_prod.zip using wget to the filepath
        os.system('wget https://zenodo.org/records/10520198/files/ckp.cl51.cloud_prod.zip -P '+filepath)
        # unzip the file
        os.system('unzip '+filepath+'ckp.cl51.cloud_prod.zip -d '+filepath)
        # remove the zip file
        os.system('rm '+filepath+'ckp.cl51.cloud_prod.zip')
        # move 2021/ 2022/ and 2023/ folders into a new ceilometer folder
        os.system('mkdir '+filepath+'ceilometer')
        os.system('mv '+filepath+'2021/'+filepath+'ceilometer/')
        os.system('mv '+filepath+'2022/'+filepath+'ceilometer/')
        os.system('mv '+filepath+'2023/'+filepath+'ceilometer/')
    else:
        print('Download the data from https://zenodo.org/records/10520198/files/ckp.cl51.cloud_prod.zip')
else:
    print('Data already downloaded')
    # we'll start by loading in one file and looking at the data
    filepath = '/storage/dlhogan/synoptic_sublimation/splash_data/ceilometer/*'
    files = glob.glob(filepath)


# %%
def process_ceilometer_data(files):
    concatenated_data = []
    
    for file in files:
        try:
            ds = xr.open_dataset(file)
            
            # Filter out range values greater than 4000
            ds = ds.where(ds.range < 4000, drop=True)
            
            # Convert backscatter profile units
            ds['backscatter_profile'] = ds['backscatter_profile'] * 1e6  # Convert from 10e-9 to 10e-3 (10e-5 m^-1 sr^-1)
            ds['backscatter_profile'].attrs['units'] = '10e-5 m^-1 sr^-1'  # Update units attribute
            
            # remove the instrumen_reported_time variable
            ds = ds.drop('instrument_reported_time')
            # Add modified dataset to list
            concatenated_data.append(ds)
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Concatenate all datasets into a single dataset
    combined_ds = xr.concat(concatenated_data, dim='time')
    
    return combined_ds

# Example usage for the given files
files_21 = glob.glob(filepath+'2021/*')
files_22 = glob.glob(filepath+'2022/*')
files_23 = glob.glob(filepath+'2023/*')

# Concatenate datasets for each year
concatenated_data_21 = process_ceilometer_data(files_21)
concatenated_data_22 = process_ceilometer_data(files_22)
concatenated_data_23 = process_ceilometer_data(files_23)

# Optionally, concatenate across all years if needed
concatenated_data_all = xr.concat([concatenated_data_21, concatenated_data_22, concatenated_data_23], dim='time')


# %%
# Example chunking your data with dask
ds = concatenated_data_all.chunk({'time': 'auto'})

# Coarsen and compute 5-minute averages using dask, with 'boundary' set to 'trim'
ds_coarsened = ds.coarsen(time=5, boundary='trim').mean()

# Calculate the time coordinates to represent the start of each trim period
# The start of each trim period is simply the existing time coordinates of the original dataset
trimmed_time_coords = ds.time[:-4:5].values

# Assign the calculated time coordinates to the coarsened dataset
ds_coarsened['time'] = trimmed_time_coords

# Compute the result using dask
ds_coarsened = ds_coarsened.compute()

# %%
# sort the dataset by time
ds_coarsened = ds_coarsened.sortby('time')

save_file = True
if save_file:
    ds_coarsened.to_netcdf('/storage/dlhogan/synoptic_sublimation/splash_data/ceilometer/splash_kp_ceilometer.nc')
# print the size of the file in MB
print(f"File size: {os.path.getsize('/storage/dlhogan/synoptic_sublimation/splash_data/ceilometer/splash_kp_ceilometer.nc') / 1e6} MB")


