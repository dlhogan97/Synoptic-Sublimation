import cdsapi
import os
import xarray as xr

# Pull ERA5 data from the CDS API client
year = input('Enter year to download:')

if not os.path.exists(f'/storage/dlhogan/data/raw_data/ERA5_reanalysis_{year}.nc'):
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "year": [year],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "pressure_level": ["500", "700"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [60, -140, 20, -100]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request,f'/storage/dlhogan/data/raw_data/ERA5_reanalysis_{year}.nc')
else:
    xr.open_dataset(f'/storage/dlhogan/data/raw_data/ERA5_reanalysis_{year}.nc')