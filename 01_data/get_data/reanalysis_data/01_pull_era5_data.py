import cdsapi
import os
import xarray as xr

# Pull ERA5 data from the CDS API client
c = cdsapi.Client()
year = input('Enter year to download:')

if not os.path.exists(f'/storage/dlhogan/data/raw_data/ERA5_reanalysis_gpot_wspd_{year}.nc'):
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '250', '500', '700',
                '850',
            ],
            'year': f'{year}',
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'area': [
                50.4, -125.6, 31.9,
                -93.5,
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        },
        f'/storage/dlhogan/data/raw_data/ERA5_reanalysis_gpot_wspd_{year}.nc')
else:
    xr.open_dataset(f'/storage/dlhogan/data/raw_data/ERA5_reanalysis_gpot_wspd_{year}.nc')