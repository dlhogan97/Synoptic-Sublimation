import pandas as pd
import numpy as np

# create a function to setup a dataframe for a windrose plot in plotly
def create_windrose_df(df, wind_dir_var, wind_spd_var):
    """
    This function takes in a dataframe and wind speed and direction variables and returns a dataframe with the wind speed binned by direction
    Inputs:
        df: pandas dataframe
        wind_dir_var: string of the wind direction variable
        wind_spd_var: string of the wind speed variable
    Outputs:
        windrose_df: pandas dataframe with the wind speed binned by direction
    """
    # group by 0-2, 2-4, 4-6, 6-8, 8-10, 10-12, 12-14, and >14 m/s bins
    df['speed_bins'] = pd.cut(df[wind_spd_var], 
                                           bins=[0,2,4,6,8,10,12,14,50], 
                                           labels=['0-2','2-4','4-6','6-8','8-10','10-12','12-14','>14+'])
    # group by cardinal wind directions
    theta_labels = [
            'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 
        ]
    theta_angles = np.arange(0, 360.1, 22.5)
    df['dir_bins'] = pd.cut(df[wind_dir_var], 
                                         bins=theta_angles, 
                                         labels=theta_labels)
    windrose_df = df.groupby(['dir_bins','speed_bins']).count().dropna()
    windrose_df['direction'] = windrose_df.index.get_level_values('dir_bins')
    windrose_df['speed'] = windrose_df.index.get_level_values('speed_bins')
    windrose_df = windrose_df[
                ['direction','speed', wind_spd_var]
            ].droplevel(0).reset_index().drop('speed_bins',axis=1)
    windrose_df.rename(columns={wind_spd_var:'frequency'}, inplace=True)
    # divide frequency by the total sum as a percentage
    windrose_df['frequency'] = 100*windrose_df['frequency']/windrose_df['frequency'].sum() 
    return windrose_df