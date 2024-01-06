from datetime import datetime
import os
import pandas as pd
from synoptic.services import stations_timeseries

### Note this will not work for data older than 1-year until I get Enterprise Synoptic.
# get data
def get_synoptic_data(stations, start_date, end_date, variables):
    params = dict(
        stid=stations,
        vars=variables,
        start=start_date,
        end=end_date,
    )
    a = stations_timeseries(verbose="HIDE", **params)
    return a

if __name__ == "__main__":
    stations = input("Enter station ids: ")
    # split on the comma and add to a list
    stations = stations.split(",")
    # remove whitespace
    stations = [station.strip() for station in stations]
    # if the last entry is empty, remove it
    if stations[-1] == "":
        stations.pop()
    # enter varibles to get 
    variables = input("Enter variables: ")
    # split on the comma and add to a list
    variables = variables.split(",")
    # remove whitespace
    variables = [variable.strip() for variable in variables]
    # if the last entry is empty, remove it
    if variables[-1] == "":
        variables.pop()
    elif len(variables) == 1:
        variables = variables[0]
    # list of variables to check against
    check_variables = ["air_temp","relative_humidity","dew_point_temperature","wind_speed","wind_direction","wind_gust",
        "altimeter","pressure","snow_depth","solar_radiation","soil_temp","precip_accum","precip_accum_one_minute",
        "precip_accum_ten_minute","precip_accum_fifteen_minute","precip_accum_30_minute","precip_accum_one_hour","precip_accum_three_hour","sea_level_pressure",
        "water_temp","weather_cond_code","cloud_layer_3_code","cloud_low_symbol","cloud_mid_symbol","cloud_high_symbol","pressure_tendency","snow_accum",
        "precip_storm","road_sensor_num","road_temp","road_freezing_temp","road_surface_condition","unknown","cloud_layer_1_code",
        "cloud_layer_2_code","precip_accum_six_hour","precip_accum_24_hour","visibility","sonic_wind_direction","metar_remark","metar",
        "air_temp_high_6_hour","air_temp_low_6_hour","peak_wind_speed","fuel_temp","fuel_moisture","ceiling","sonic_wind_speed",
        "pressure_change_code","precip_smoothed","soil_temp_ir","temp_in_case","soil_moisture","volt","created_time_stamp",
        "last_modified","snow_smoothed","precip_manual","precip_accum_manual","precip_accum_5_minute_manual","precip_accum_10_minute_manual","precip_accum_15_minute_manual",
        "precip_accum_3_hour_manual","precip_accum_6_hour_manual","precip_accum_24_hour_manual","snow_accum_manual","snow_interval","road_subsurface_tmp",
        "T_water_temp","evapotranspiration","snow_water_equiv","precipitable_water_vapor","air_temp_high_24_hour","air_temp_low_24_hour","peak_wind_direction",
        "net_radiation","soil_moisture_tension","pressure_1500_meter","air_temp_wet_bulb","air_temp_2m","air_temp_10m","surface_temp","net_radiation_sw",
        "net_radiation_lw","sonic_air_temp","sonic_vertical_vel","sonic_zonal_wind_stdev","sonic_vertical_wind_stdev","sonic_air_temp_stdev","vertical_heat_flux",
        "friction_velocity","w_ratio","sonic_ob_count","sonic_warn_count","moisture_stdev","vertical_moisture_flux",
        "virtual_temp","geopotential_height","outgoing_radiation_sw","PM_25_concentration","ozone_concentration","black_carbon_concentration","columbia_river_datum",
        "CO_concentration","derived_aerosol_boundary_layer_depth","electric_conductivity","estimated_snowfall_rate","filter_percentage","fosberg_fire_weather_index",
        "gage_height","incoming_radiation_lw","incoming_radiation_uv","mean_higher_high_water",
        "mean_high_water","mean_lower_low_water","mean_low_water","mean_sea_level","mean_tide_level","NH3_concentration","NO2y_concentration",
        "NO2_concentration","north_american_vertical_datum","NOx_concentration","NOy_concentration","NO_concentration","outgoing_radiation_lw","outgoing_radiation_uv",
        "particulate_concentration","past_weather_code","permittivity","PM_10_concentration","precip_accum_12_hour","precip_accum_five_minute","precip_accum_since_00utc",
        "precip_accum_since_7_local","precip_accum_since_local_midnight","precip_interval","sensor_error_code","snow_accum_24_hour","snow_accum_since_7_local",
        "snow_core_water_equiv","SO2_concentration","station_datum","stream_flow","surface_level","synop","UV_370nm_concentration","visibility_code",
        "water_current_direction","water_current_speed","wind_cardinal_direction","wind_chill","heat_index","weather_condition",
        "weather_summary","cloud_layer_1","cloud_layer_2","cloud_layer_3","wet_bulb_temperature"]

    # drop variables that are not in the check_variables list
    variables = [variable for variable in variables if variable in check_variables]
    start = input("Enter start date (YYYY-MM-DD): ")
    end = input("Enter end date (YYYY-MM-DD): ")
    # convert start and end to datetime objects
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    
    # ask for output file path
    output_file = input("Enter output file path: ")
    # ask for output file name
    output_file_name = input("Enter output file name: ") 

    # get data 
    data = get_synoptic_data(stations, start, end, variables)

    # write data to csv if data is a pandas dataframe
    if isinstance(data, pd.DataFrame):
        print("Saving data set...")
        data.to_csv(os.path.join(output_file, output_file_name))
        print("Find data set at: {}".format(os.path.join(output_file, output_file_name)))
    else:
        print("Multiple locations provided, data set not saved.")

