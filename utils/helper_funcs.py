import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT, add_metpy_logo
from metpy.units import units
import metpy.calc as mcalc

WIND_VARIABLES = [
# Sonic Anemometer Data for 4 towers
'spd_1m_uw',     'dir_1m_uw',     'u_1m_uw',   'v_1m_uw',   'w_1m_uw',
'spd_3m_uw',     'dir_3m_uw',     'u_3m_uw',   'v_3m_uw',   'w_3m_uw',
'spd_10m_uw',    'dir_10m_uw',    'u_10m_uw',  'v_10m_uw',  'w_10m_uw',
'spd_1m_ue',     'dir_1m_ue',     'u_1m_ue',   'v_1m_ue',   'w_1m_ue',
'spd_3m_ue',     'dir_3m_ue',     'u_3m_ue',   'v_3m_ue',   'w_3m_ue',
'spd_10m_ue',    'dir_10m_ue',    'u_10m_ue',  'v_10m_ue',  'w_10m_ue',
'spd_1m_d',     'dir_1m_d',     'u_1m_d',   'v_1m_d',   'w_1m_d',
'spd_3m_d',     'dir_3m_d',     'u_3m_d',   'v_3m_d',   'w_3m_d',
'spd_10m_d',    'dir_10m_d',    'u_10m_d',  'v_10m_d',  'w_10m_d',
'spd_2m_c',     'dir_2m_c',     'u_2m_c',   'v_2m_c',   'w_2m_c',
'spd_3m_c',     'dir_3m_c',     'u_3m_c',   'v_3m_c',   'w_3m_c',
'spd_5m_c',     'dir_5m_c',     'u_5m_c',   'v_5m_c',   'w_5m_c',
'spd_10m_c',    'dir_10m_c',    'u_10m_c',  'v_10m_c',  'w_10m_c',
'spd_15m_c',    'dir_15m_c',    'u_15m_c',  'v_15m_c',  'w_15m_c',
'spd_20m_c',    'dir_20m_c',    'u_20m_c',  'v_20m_c',  'w_20m_c',
]
COUNT_VARIABLES = ['counts_3m_c',
 'counts_3m_ue',
 'counts_1m_d',
 'counts_15m_c',
 'counts_10m_uw',
 'counts_1m_ue',
 'counts_20m_c',
 'counts_2m_c',
 'counts_10m_ue',
 'counts_1m_c',
 'counts_1m_uw',
 'counts_3m_uw',
 'counts_10m_c',
 'counts_3m_d',
 'counts_5m_c',
 'counts_10m_d']

TURBULENT_FLUX_VARIABLES = ['Hs_1m_uw','Hl_webb_1m_uw',
                            'Hs_3m_uw','Hl_webb_3m_uw',
                            'Hs_10m_uw','Hl_webb_10m_uw',
                            
                            'Hs_1m_ue','Hl_webb_1m_ue',
                            'Hs_3m_ue','Hl_webb_3m_ue',
                            'Hs_10m_ue','Hl_webb_10m_ue',
                            
                            'Hs_1m_d','Hl_webb_1m_d',
                            'Hs_3m_d','Hl_webb_3m_d',
                            'Hs_10m_d','Hl_webb_10m_d',
                            
                            'Hs_2m_c','Hl_webb_2m_c',
                            'Hs_3m_c','Hl_webb_3m_c',
                            'Hs_5m_c','Hl_webb_5m_c',
                            'Hs_10m_c','Hl_webb_10m_c',
                            'Hs_15m_c','Hl_webb_15m_c',
                            'Hs_20m_c','Hl_webb_20m_c',]
TURBULENCE_VARIABLES = [
    'tc_1m_uw',        'u_u__1m_uw',    'v_v__1m_uw',    'w_w__1m_uw',    
        'u_w__1m_uw',    'v_w__1m_uw',  'u_tc__1m_uw',  'v_tc__1m_uw',   'u_h2o__1m_uw',  'v_h2o__1m_uw',   'w_tc__1m_uw',   'w_h2o__1m_uw', 'w_t_1m_uw', 'w_mr_1m_uw',
    'tc_3m_uw',        'u_u__3m_uw',    'v_v__3m_uw',    'w_w__3m_uw',    
        'u_w__3m_uw',    'v_w__3m_uw',  'u_tc__3m_uw',  'v_tc__3m_uw',   'u_h2o__3m_uw',  'v_h2o__3m_uw',   'w_tc__3m_uw',   'w_h2o__3m_uw', 'w_t_3m_uw', 'w_mr_3m_uw',
    'tc_10m_uw',      'u_u__10m_uw',   'v_v__10m_uw',   'w_w__10m_uw',   
        'u_w__10m_uw',   'v_w__10m_uw', 'u_tc__10m_uw', 'v_tc__10m_uw',  'u_h2o__10m_uw', 'v_h2o__10m_uw',  'w_tc__10m_uw',  'w_h2o__10m_uw', 'w_t_10m_uw', 'w_mr_10m_uw',

    'tc_1m_ue',        'u_u__1m_ue',    'v_v__1m_ue',    'w_w__1m_ue',    
        'u_w__1m_ue',    'v_w__1m_ue',  'u_tc__1m_ue',  'v_tc__1m_ue',   'u_h2o__1m_ue',  'v_h2o__1m_ue',   'w_tc__1m_ue',   'w_h2o__1m_ue', 'w_t_1m_ue', 'w_mr_1m_ue',
    'tc_3m_ue',        'u_u__3m_ue',    'v_v__3m_ue',    'w_w__3m_ue',    
        'u_w__3m_ue',    'v_w__3m_ue',  'u_tc__3m_ue',  'v_tc__3m_ue',   'u_h2o__3m_ue',  'v_h2o__3m_ue',   'w_tc__3m_ue',   'w_h2o__3m_ue', 'w_t_3m_ue', 'w_mr_3m_ue',
    'tc_10m_ue',      'u_u__10m_ue',   'v_v__10m_ue',   'w_w__10m_ue',   
        'u_w__10m_ue',   'v_w__10m_ue', 'u_tc__10m_ue', 'v_tc__10m_ue',  'u_h2o__10m_ue', 'v_h2o__10m_ue',  'w_tc__10m_ue',  'w_h2o__10m_ue', 'w_t_10m_ue', 'w_mr_10m_ue',

    'tc_1m_d',         'u_u__1m_d',    'v_v__1m_d',    'w_w__1m_d',    
        'u_w__1m_d',    'v_w__1m_d',  'u_tc__1m_d',  'v_tc__1m_d',   'u_h2o__1m_d',  'v_h2o__1m_d',   'w_tc__1m_d',   'w_h2o__1m_d', 'w_t_1m_d', 'w_mr_1m_d',
    'tc_3m_d',         'u_u__3m_d',    'v_v__3m_d',    'w_w__3m_d',    
        'u_w__3m_d',    'v_w__3m_d',  'u_tc__3m_d',  'v_tc__3m_d',   'u_h2o__3m_d',  'v_h2o__3m_d',   'w_tc__3m_d',   'w_h2o__3m_d', 'w_t_3m_d', 'w_mr_3m_d',
    'tc_10m_d',       'u_u__10m_d',   'v_v__10m_d',   'w_w__10m_d',   
        'u_w__10m_d',   'v_w__10m_d', 'u_tc__10m_d', 'v_tc__10m_d',  'u_h2o__10m_d', 'v_h2o__10m_d',  'w_tc__10m_d',  'w_h2o__10m_d', 'w_t_10m_d', 'w_mr_10m_d',

    'tc_2m_c',     'u_u__2m_c',    'v_v__2m_c',    'w_w__2m_c',    
        'u_w__2m_c',    'v_w__2m_c',  'u_tc__2m_c',  'v_tc__2m_c',   'u_h2o__2m_c',  'v_h2o__2m_c',   'w_tc__2m_c',   'w_h2o__2m_c', 'w_t_2m_c', 'w_mr_2m_c',
    'tc_3m_c',     'u_u__3m_c',    'v_v__3m_c',    'w_w__3m_c',    
        'u_w__3m_c',    'v_w__3m_c',  'u_tc__3m_c',  'v_tc__3m_c',   'u_h2o__3m_c',  'v_h2o__3m_c',   'w_tc__3m_c',   'w_h2o__3m_c', 'w_t_3m_c', 'w_mr_3m_c',
    'tc_5m_c',     'u_u__5m_c',    'v_v__5m_c',    'w_w__5m_c',    
        'u_w__5m_c',    'v_w__5m_c',  'u_tc__5m_c',  'v_tc__5m_c',   'u_h2o__5m_c',  'v_h2o__5m_c',   'w_tc__5m_c',   'w_h2o__5m_c', 'w_t_5m_c', 'w_mr_5m_c',
    'tc_10m_c',   'u_u__10m_c',   'v_v__10m_c',   'w_w__10m_c',   
        'u_w__10m_c',   'v_w__10m_c', 'u_tc__10m_c', 'v_tc__10m_c',  'u_h2o__10m_c', 'v_h2o__10m_c',  'w_tc__10m_c',  'w_h2o__10m_c', 'w_t_10m_c', 'w_mr_10m_c',
    'tc_15m_c',   'u_u__15m_c',   'v_v__15m_c',   'w_w__15m_c',   
        'u_w__15m_c',   'v_w__15m_c', 'u_tc__15m_c', 'v_tc__15m_c',  'u_h2o__15m_c', 'v_h2o__15m_c',  'w_tc__15m_c',  'w_h2o__15m_c', 'w_t_15m_c', 'w_mr_15m_c',
    'tc_20m_c',   'u_u__20m_c',   'v_v__20m_c',   'w_w__20m_c',   
        'u_w__20m_c',   'v_w__20m_c', 'u_tc__20m_c', 'v_tc__20m_c',  'u_h2o__20m_c', 'v_h2o__20m_c',  'w_tc__20m_c',  'w_h2o__20m_c', 'w_t_20m_c', 'w_mr_20m_c',
]
WATER_VAPOR_VARIABLES = [
'h2o_1m_uw', 'h2o_3m_uw', 'h2o_10m_uw', 'h2o_1m_ue', 'h2o_3m_ue', 'h2o_10m_ue', 'h2o_1m_d', 'h2o_3m_d', 'h2o_10m_d', 'h2o_2m_c', 'h2o_3m_c', 'h2o_5m_c', 'h2o_10m_c', 'h2o_15m_c', 'h2o_20m_c'
]
TEMPERATURE_VARIABLES = [    
    # Temperature & Relative Humidity Array 
    'T_2m_c', 'T_3m_c', 'T_4m_c', 'T_5m_c', 'T_6m_c', 'T_7m_c', 'T_8m_c', 'T_9m_c', 'T_10m_c',
    'T_11m_c', 'T_12m_c', 'T_13m_c', 'T_14m_c', 'T_15m_c', 'T_16m_c', 'T_17m_c', 'T_18m_c', 'T_19m_c', 'T_20m_c',

    'RH_2m_c', 'RH_3m_c', 'RH_4m_c', 'RH_5m_c', 'RH_6m_c', 'RH_7m_c', 'RH_8m_c', 'RH_9m_c', 'RH_10m_c',
    'RH_11m_c','RH_12m_c','RH_13m_c','RH_14m_c','RH_15m_c','RH_16m_c','RH_17m_c','RH_18m_c','RH_19m_c','RH_20m_c'
]
PRESSURE_VARIABLES = [
    # Pressure Sensors
    'P_20m_c',
    'P_10m_c', 'P_10m_d', 'P_10m_uw', 'P_10m_ue'
]
SNOW_FLUX = [
    # Blowing snow/FlowCapt Sensors
    'SF_avg_1m_ue', 'SF_avg_2m_ue',
]
RADIAION_VARIABLES = [
    'Rpile_out_9m_d', 'Rpile_in_9m_d', 'Rsw_in_9m_d', 'Rsw_out_9m_d'
]
# create a function to setup a dataframe for a windrose plot in plotly
def create_windrose_df(
    df_or_speed,
    wind_dir_var: str | None = "wind_dir",
    wind_spd_var: str | None = "wind_spd",
    *,
    dir_bin_size: float = 22.5,               # degrees per sector (22.5 ° → 16‑point rose)
    speed_bins: tuple | list = (0, 2, 4, 6, 8, 10, 12, 14, np.inf),
    speed_labels: list | None = None,         # e.g. ["0–2", "2–4", …]  (optional)
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Return a tidy wind‑rose dataframe with one row per (direction sector, speed bin).

    Parameters
    ----------
    df_or_speed : pandas.DataFrame | array‑like
        Either a DataFrame containing *wind_dir_var* and *wind_spd_var* columns
        **or** any array‑like of speeds (if so, *wind_dir_var*/*wind_spd_var*
        are ignored and you must pass both `speed` and `direction` separately).
    wind_dir_var, wind_spd_var : str, optional
        Column names for wind direction (degrees 0‑360) and speed.
    dir_bin_size : float, optional
        Width of directional sectors in degrees (default 22.5 ° → 16 sectors).
    speed_bins : sequence, optional
        Bin edges for speed categories.  Last edge may be *np.inf*.
    speed_labels : list, optional
        Custom labels for the speed categories (must match len(speed_bins)‑1).
        If *None*, pandas IntervalIndex labels are kept.
    drop_na : bool, optional
        Drop rows with NA directions/speeds before binning.

    Returns
    -------
    pandas.DataFrame with columns:
        * direction   – categorical sector label
        * speed       – categorical speed bin label (Interval or supplied label)
        * frequency   – count of samples in that cell
        * frequency_pct – relative frequency (%) of that cell
    """
    # ------------------------------------------------------------------ 1. Parse inputs
    if isinstance(df_or_speed, pd.DataFrame):
        speed = df_or_speed[wind_spd_var]
        direction = df_or_speed[wind_dir_var]
    else:
        raise ValueError(
            "Pass a DataFrame containing speed & direction columns "
            "for the simplest usage."
        )

    if drop_na:
        mask = speed.notna() & direction.notna()
        speed, direction = speed[mask], direction[mask]

    # ------------------------------------------------------------------ 2. Direction sectors
    # Ensure all directions are 0 ≤ dir < 360
    direction = direction % 360                      # ensure 0 ≤ dir < 360
    half = dir_bin_size / 2
    dir_edges = np.arange(-half, 360 + half, dir_bin_size)

    # One label per interval: centre angle of each sector (e.g. 0°, 30°, …)
    sector_centres = (dir_edges[:-1] + dir_edges[1:]) / 2
    sector_labels  = (sector_centres % 360).astype(int)   # wrap 360→0

    direction_cat = pd.cut(
        direction,
        bins=dir_edges,
        labels=sector_labels,        # now len(labels) == len(edges)‑1 ✔
        ordered=True,
        right=False,                 # left‑inclusive, right‑exclusive
        include_lowest=True,
    )

    # ------------------------------------------------------------------ 3. Speed bins
    speed_cat = pd.cut(speed, bins=speed_bins, labels=speed_labels, right=True)

    # ------------------------------------------------------------------ 4. Frequency table
    rose = (
        pd.DataFrame({"direction": direction_cat, "speed": speed_cat})
        .groupby(["direction", "speed"], observed=True)
        .size()
        .rename("frequency")
        .reset_index()
    )

    # percentage frequency
    rose["frequency_pct"] = rose["frequency"] / rose["frequency"].sum() * 100

    return rose

def simple_sounding(ds):
    """
    This function takes in a dataset and plots a skew-t diagram with the radiosonde data
    Inputs:
        ds: xarray dataset
    Outputs:
        fig: matplotlib figure
    """
    # get the first time index and save as a string with format YYYY-MM-DD HH:MM
    time = pd.to_datetime(ds['time'].values[0]).strftime('%Y-%m-%d %H:%M')
    # check if tdew a variable in the dataset
    if 'tdew' not in ds:
        # if not, calculate it
        ds['tdew'] = mcalc.dewpoint_from_relative_humidity(ds['tdry'],ds['rh'])
    # get index for values of p > 200
    ix = np.where(ds['pres'].values > 200)[0]
    p = ds['pres'].values[ix] * units.hPa
    T = ds['tdry'].values[ix] * units.degC
    Td = ds['tdew'].values[ix] * units.degC
    u = ds['u_wind'].values[ix] * units('m/s')
    v = ds['v_wind'].values[ix] * units('m/s')
    # find the pressures where T is between -12 and -18
    ix_dgz = np.where((T > -18 * units.degC) & (T < -12 * units.degC))[0]
    fig = plt.figure(figsize=(8, 12))
    # increase whitespace at the bottom of the plot
    fig.subplots_adjust(bottom=0.2)
    # Example of defining your own vertical barb spacing
    skew = SkewT(fig, aspect=100)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    temp = skew.plot(p, T, 'r', label='Temperature')
    tdew = skew.plot(p, Td, 'g', label='Dew Point')
    # change the color and linestyle of the grid lines
    isotherm=skew.ax.grid(True, which='major', axis='both', color='white', linestyle='-', linewidth=1, alpha=0.5,label='Isotherms')
    # Set some better labels than the default
    skew.ax.set_xlabel('Temperature (\N{DEGREE CELSIUS})')
    skew.ax.set_ylabel('Pressure (mb)')

    # Set spacing interval--Every 50 mb from 1000 to 100 mb
    my_interval = np.arange(200, 720, 50) * units('mbar')

    # Get indexes of values closest to defined interval
    ix = mcalc.resample_nn_1d(p, my_interval)

    # Plot only values nearest to defined interval values
    barbs = skew.plot_barbs(p[ix], u[ix], v[ix], color='white')

    # Add the relevant special lines
    dry_adiabats = skew.plot_dry_adiabats(colors='red',alpha=0.5, linestyle='-', label='Dry Adiabats')
    moist_adiabats = skew.plot_moist_adiabats(colors='blue', alpha=0.75, linestyle='-', label='Moist Adiabats')
    mixing_ratios = skew.plot_mixing_lines(colors='grey',alpha=0.5, label='Mixing Ratio')
    skew.ax.set_ylim(p[0], 200)

    # plot a yellow line at the top and bottom of the dgz only on the left 0.25 of the plot
    skew.ax.axhline(y=p[ix_dgz[0]], color='yellow', linestyle='--', alpha=0.5, xmin=0, xmax=0.25)
    skew.ax.axhline(y=p[ix_dgz[-1]], color='yellow', linestyle='--', alpha=0.5, xmin=0, xmax=0.25)
    # label the zone DGZ
    skew.ax.text(T.min().magnitude-5, p[ix_dgz[0]].magnitude-30, 'DGZ', color='yellow', alpha=0.5)

    # set xaxis values to between min and max values + 10
    skew.ax.set_xlim(T.min().magnitude - 10, T.max().magnitude + 10)

    # make the outline of the figure white
    skew.ax.spines['top'].set_color('white')
    skew.ax.spines['left'].set_color('white')
    skew.ax.spines['right'].set_color('white')
    skew.ax.spines['bottom'].set_color('white') 
    # make the background color black
    skew.ax.set_facecolor('black')
    # make the whole figure black
    fig.patch.set_facecolor('black')
    # make xaxis ticks, labels, and ticklabels white
    skew.ax.xaxis.set_tick_params(color='white')
    skew.ax.xaxis.label.set_color('white')
    skew.ax.tick_params(axis='x', colors='white')
    # make yaxis ticks, labels, and ticklabels white
    skew.ax.yaxis.set_tick_params(color='white')
    skew.ax.yaxis.label.set_color('white')
    skew.ax.tick_params(axis='y', colors='white')
    # make the title white
    skew.ax.set_title(f'Radiosonde Sounding for {time} UTC', color='white')
    # add legend outside the plot on the right
    h, l = skew.ax.get_legend_handles_labels()
    
    skew.ax.legend(
                   loc='center left', bbox_to_anchor=(1.05, 0.5),
                   facecolor='black', labelcolor='white')
    # add metpy logo as an inset to the bottom left corner
    logo_fig= plt.gcf()
    add_metpy_logo(logo_fig, 750, -p.max().magnitude+1100, size='small', zorder=0)
    return fig

def mean_sounding(df_mean, title):
    """
    This function takes in a dataframe of mean u,v wind components, mean temperature, mean dewpoint, and mean pressure at 10 mb intervals
    and plots a skew-t diagram with the mean radiosonde data.
    Inputs:
        df_mean: pandas dataframe
        title: string of the title for the plot
    Outputs:
        fig: matplotlib figure
    """
    # get the mean values
    p = df_mean['pres'].values * units.hPa
    T = df_mean['tdry'].values * units.degC
    Td = df_mean['tdew'].values * units.degC
    u = df_mean['u_wind'].values * units('m/s')
    v = df_mean['v_wind'].values * units('m/s')
    # find the pressures where T is between -12 and -18
    ix_dgz = np.where((T > -18 * units.degC) & (T < -12 * units.degC))[0]
    fig = plt.figure(figsize=(8, 12))
    # increase whitespace at the bottom of the plot
    fig.subplots_adjust(bottom=0.2)
    # Example of defining your own vertical barb spacing
    skew = SkewT(fig, aspect=100)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    temp = skew.plot(p, T, 'r', label='Temperature')
    tdew = skew.plot(p, Td, 'g', label='Dew Point')
    # change the color and linestyle of the grid lines
    isotherm=skew.ax.grid(True, which='major', axis='both', color='white', linestyle='-', linewidth=1, alpha=0.5,label='Isotherms')
    # Set some better labels than the default
    skew.ax.set_xlabel('Temperature (\N{DEGREE CELSIUS})')
    skew.ax.set_ylabel('Pressure (mb)')

    # Set spacing interval--Every 50 mb from 1000 to 100 mb
    my_interval = np.arange(200, 720, 50) * units('mbar')

    # Get indexes of values closest to defined interval
    ix = mcalc.resample_nn_1d(p, my_interval)

    # Plot only values nearest to defined interval values
    barbs = skew.plot_barbs(p[::5], u[::5], v[::5], color='white')

    # Add the relevant special lines
    dry_adiabats = skew.plot_dry_adiabats(colors='red',alpha=0.5, linestyle='-', label='Dry Adiabats')
    moist_adiabats = skew.plot_moist_adiabats(colors='blue', alpha=0.75, linestyle='-', label='Moist Adiabats')
    mixing_ratios = skew.plot_mixing_lines(colors='grey',alpha=0.5, label='Mixing Ratio')
    skew.ax.set_ylim(p[0], 200)

    # plot a yellow line at the top and bottom of the dgz only on the left 0.25 of the plot
    skew.ax.axhline(y=p[ix_dgz[0]], color='yellow', linestyle='--', alpha=0.5, xmin=0, xmax=0.25)
    skew.ax.axhline(y=p[ix_dgz[-1]], color='yellow', linestyle='--', alpha=0.5, xmin=0, xmax=0.25)
    # label the zone DGZ
    skew.ax.text(T.min().magnitude-5, p[ix_dgz[0]].magnitude-30, 'DGZ', color='yellow', alpha=0.5)

    # set xaxis values to between min and max values + 10
    skew.ax.set_xlim(T.min().magnitude - 10, T.max().magnitude + 10)

    # make the outline of the figure white
    skew.ax.spines['top'].set_color('white')
    skew.ax.spines['left'].set_color('white')
    skew.ax.spines['right'].set_color('white')
    skew.ax.spines['bottom'].set_color('white') 
    # make the background color black
    skew.ax.set_facecolor('black')
    # make the whole figure black
    fig.patch.set_facecolor('black')
    # make xaxis ticks, labels, and ticklabels white
    skew.ax.xaxis.set_tick_params(color='white')
    skew.ax.xaxis.label.set_color('white')
    skew.ax.tick_params(axis='x', colors='white')
    # make yaxis ticks, labels, and ticklabels white
    skew.ax.yaxis.set_tick_params(color='white')
    skew.ax.yaxis.label.set_color('white')
    skew.ax.tick_params(axis='y', colors='white')
    # make the title white
    skew.ax.set_title(title, color='white')
    # add legend outside the plot on the right
    h, l = skew.ax.get_legend_handles_labels()
    
    skew.ax.legend(
                   loc='center left', bbox_to_anchor=(1.05, 0.5),
                   facecolor='black', labelcolor='white')
    # add metpy logo as an inset to the bottom left corner
    logo_fig= plt.gcf()
    add_metpy_logo(logo_fig, 750, -p.max().magnitude+1100, size='small', zorder=0)
    return 

def adjust_wdir(df):
    """
    This function takes in a dataframe and calculates the wind direction from the u and v wind components
    Inputs:
        df: pandas dataframe
    Outputs:
        df: pandas dataframe with the wind direction calculated
    """
    # get all the variables with 'dir' in the name
    
    dir_vars = df.filter(regex='dir_*').columns
    if len(dir_vars) == 0:
        print('No wind direction variables found')
        return
    else:
        # for each u_* and v_* variable, calculate the wind direction
        for dir_var in dir_vars:
            # for the column, extract everything after the first underscore
            loc = dir_var.split('_', 1)[1]
            u_loc = 'u_' + loc
            v_loc = 'v_' + loc
            # create a column with the wind direction
            df[dir_var] = np.arctan2(df[v_loc], df[u_loc]) * 180 / np.pi
            # convert wind direction to compass direction
            df[dir_var] = (270 - df[dir_var]) % 360
        return df

VARIABLES = ['atmos_pressure','temp_mean', 'rh_mean','vapor_pressure_mean','wspd_arith_mean','wdir_vec_mean']
VARIABLES_KP = ['mean_u','mean_v','mean_t','mean_q','atm_pres','wind_spd','wind_dir','lv']
def qc_sail_met(ds, variables, temp_var, rh_var, wspd_var, vapor_pressure_var):
    # first let's add qc to all the variables as a new list
    qc_variables = ['qc_'+v for v in variables]
    ds_qc = ds[variables+qc_variables]
    # now if any qc variable is not 0, we will set the variable to nan
    for v in variables:
        ds_qc[v] = ds_qc[v].where(ds_qc['qc_'+v]==0, other=np.nan)
    # now let's drop the qc variables
    ds_qc = ds_qc.drop_vars(qc_variables)
    # now let's set extreme values for certain variables
    temp_max = 30
    temp_min = -40
    rh_min = 0
    wspd_max = 40 # max in m/s
    vapor_pressure_max = 7.835 # this is the saturation vapor pressure at 40C
    # now let's filter out the extremes and fill with nan
    ds_qc[temp_var] = ds_qc[temp_var].where((ds_qc[temp_var]<temp_max) & (ds_qc[temp_var]>temp_min), other=np.nan)
    if rh_var == 'rh_mean':
        ds_qc['rh_mean'] = ds_qc['rh_mean'].where(ds_qc['rh_mean']>rh_min, other=np.nan)
    ds_qc[wspd_var] = ds_qc[wspd_var].where(ds_qc[wspd_var]<wspd_max, other=np.nan)
    ds_qc[vapor_pressure_var] = ds_qc[vapor_pressure_var].where(ds_qc[vapor_pressure_var]<vapor_pressure_max, other=np.nan)
    # count how many data points are nan for each variable and print that out
    for v in variables:
        print(f'{v} has {ds_qc[v].isnull().sum().values} nan values')
    print('-------------------')
    # return the qc'd dataset
    return ds_qc

def localize_time(data, current_time_zone='UTC', local_time_zone='MST'):
    import pandas as pd
    import xarray as xr
    """
    This function takes in a dataset or dataframe and localizes the time to MST
    """
    if isinstance(data, xr.Dataset):        
        data['time'] = pd.to_datetime(data.time).tz_localize(current_time_zone).tz_convert(local_time_zone).tz_localize(None)  
    elif isinstance(data, pd.DataFrame):
        data.index = pd.to_datetime(data.index).tz_localize(current_time_zone).tz_convert(local_time_zone).tz_localize(None)
    else:
        raise ValueError('Data must be either an xarray dataset or a pandas dataframe')
    return data
