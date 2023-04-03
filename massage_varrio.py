# -*- coding: utf-8 -*-
"""
Sript for processing Värriö-dataset

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import get_avaa_dataset

def plot_timeseries(dat, variables):
    """
    plots timeseries, waits for user input to proceed
    Args:
        dat - dataframe
        variables (str or list of str) - columns
    Returns:
        none
    """
    
    if variables is str:
        variables = list(variables)
    
    
    for v in variables:
        fig, ax = plt.subplots()
        ax.plot(dat[v], '-')
        ax.set_title(v)
        plt.show()
        
        w = plt.waitforbuttonpress()
        
        #ax.clear()

#%% Step 1: Read datafiles, compile dataframe with desired variables & units.
        
ffile = [r'c:\DATA\GFB\FI-Var\varrio_flux.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_flux2.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_flux_anc.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_met1.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_met2.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_met3.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_radi.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_soil.txt',\
         r'c:\DATA\GFB\FI-Var\varrio_pressure.txt'
        ]

raw = get_avaa_dataset(ffile, UTC=True)

cols = list(raw.columns)

# ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 
# 'VAR_EDDY.u_star', 
newcols = [s.split('.')[-1] for s in cols]
print(newcols)
raw.columns = newcols

raw['Reco'] = raw['NEE'] + raw['GPP']

flx = raw[['Year', 'Month', 'Day', 'Hour', 'Minute',
           'NEE', 'GPP', 'Reco', 'H', 'LE', 'G', 'NET', 'u_star', 'U', 'wind_dir', 'tau', 'MO_length',
           'TDRY3', 'HUM_RH', 'av_c', 'P', 'Precipacc', 'rainint', 'snowint',
           'GLOB', 'PAR', 'RPAR', 'diffPAR', 'LWin2', 'UVA', 'UVB',
           'ST', 'Tsoil', 'Tsoil5', 'Tsoil25', 'wsoil', 'wsoil5', 'wsoil25',
           'SNOWDEPTH', 
           'F_c', 'H_storage_flux', 'LE_storage_flux', 'Qc_gapf_NEE', 'Qc_H', 'Qc_LE',
           # sub-canopy fluxes
           #'F_c_sub', 'H_sub', 'LE_sub', 'u_star_sub', 'U_sub', 'wind_dir_sub', 'tau_sub', 'MO_length_sub',
           #'av_t_sub', 'std_w_sub'
          ]]

# convert units
flx['P'] *= 100 # Pa
flx['SNOWDEPTH'] /= 100 # m
flx['SNOWDEPTH'] = np.maximum(flx['SNOWDEPTH'], 0.0)
flx['Precipacc'] *= 30 # mm, accumulated
flx['rainint'] *= 0.5 # mm, accumulated
flx['snowint'] *=0.5 # mm, accumulated
flx['F_c'][flx['Qc_gapf_NEE'] > 0] = np.NaN

# convert variable names
flx.rename(columns={'NET': 'Rn', 'TDRY3': 'Tair', 'HUM_RH': 'RH', 'av_c': 'CO2',
                    'Precipacc': 'Prec', 'rainint': 'Rainacc', 'snowint': 'Snowacc',
                    'GLOB': 'Rg', 'PAR': 'Par', 'diffPAR': 'diffPar', 'RPAR': 'rPar', 'LWin2': 'LWin',
                    'ST': 'Tsoil0', 'Tsoil': 'Tsoil1', 'Qc_gapf_NEE': 'Qc_NEE',
                    'SNOWDEPTH': 'Snowdepth'},
           inplace=True)

#sub-canopy fluxes
flx_sub = raw[['Year', 'Month', 'Day', 'Hour', 'Minute',
              'F_c_sub', 'H_sub', 'LE_sub', 'u_star_sub', 'U_sub', 'wind_dir_sub', 'tau_sub', 'MO_length_sub',
          'av_t_sub', 'std_w_sub']]

flx_sub.rename(columns={'av_t_sub': 'Tair_sub'}, inplace = True)


#%% Get FMI data
download = False

if download:
    from src.utils import RetrieveFMIObs
    
    #def RetrieveFMIObs(latin, lonin, ts, te, deltat, distlm=50000, interpolateflag=False,varsin=None, dattype='weather'):
    lat = 67.747
    lon = 29.618
    
    ts = '2013-01-01'
    te = '2022-12-31'
    freq = '30min'
    
    fmi, fmiunits, fmisite =  RetrieveFMIObs(lat, lon, ts, te, freq, resample=True)
    
    fmi = fmi[['Ta_FMI', 'RH_FMI', 'P_RAININTENSITY_FMI', 'D_SNOW_FMI']]
    
    fmi = fmi.reindex_like(flx)
    fmi.to_csv(r'c:\DATA\GFB\FI-Var\FMI_obs.dat', sep=';',
              float_format='%.2f', na_rep='NaN', index=True, index_label='Date')

else:
    fmi = pd.read_csv(r'c:\DATA\GFB\FI-Var\FMI_obs.dat', sep=';')
    fmi.index = pd.DatetimeIndex(fmi['Date'])
    fmi.drop('Date', axis=1, inplace=True)
    fmi.rename(columns={'TA_FMI': 'Tair', 'RH_FMI': 'RH', 'P_RAININTENSITY_FMI': 'Prec',
                        'D_SNOW_FMI': 'Snowdepth'}, inplace=True)
    fmi['Prec'] /= 2 # mm/30min
    
    fmi = fmi.reindex(flx.index)

#%% Create model forcing file

met = flx.copy()
met.rename(columns={'Tsoil1':'Tsoil', 'wsoil': 'Wsoil'}, inplace=True)
met = met[['Rn', 'Rg', 'Par', 'diffPar', 'U', 'u_star', 'Tair', 'RH', 'CO2', 'LWin', 'P', 'Prec',
           'Tsoil', 'Wsoil', 'Snowdepth']]

met = met[~met.index.duplicated(keep='first')]

from src.utils import create_forcingfile, save_df_to_csv
timezone = +2.0
lat = 67.747
lon = 29.618
forc, flags, readme = create_forcingfile(met.copy(), 'output_file',lat, lon, met_data=fmi.copy(), timezone=timezone,
                                 CO2_constant=False, short_gap_len=5)
forc.insert(0, 'Year', forc.index.year.values)
forc.insert(1, 'Month', forc.index.month.values)
forc.insert(2, 'Day', forc.index.day.values)
forc.insert(3, 'Hour', forc.index.hour.values)
forc.insert(4, 'Minute', forc.index.minute.values)

c = list(flags.columns)

cc = ['flag_' + k for k in c]
flags.columns = cc
flags.index=forc.index
forc = pd.concat([forc, flags], axis=1)

readme +="\nflag_x: 0 = observed, -1 = filled by another instrument, 1 = linear interpolation, 2 > MDV (days), \n\
            -2 = Prec set to 0.0 in absence of data"


#%%  *** save to asciii files

ix = list(flx.columns).index('Snowdepth')
flx.insert(ix+1,'Snowdepth2', fmi['Snowdepth'])
ix = list(flx.columns).index('Snowacc')
flx.insert(ix+1, 'Prec_FMI', fmi['Prec'])


flx = pd.concat([flx, flx_sub[['F_c_sub', 'H_sub', 'LE_sub', 'u_star_sub', 'U_sub', 
                               'wind_dir_sub', 'tau_sub', 'MO_length_sub', 
                               'Tair_sub', 'std_w_sub']]], axis=1)

# observations
flx.to_csv(r'c:\DATA\GFB\FI-Var\Proc\FI-Var_2013_2022.dat', sep=';',
           float_format='%.2f', na_rep='NaN', index=False)

#flx_sub.to_csv(r'c:\DATA\GFB\FI-Var\Proc\FI-Var_sub_2013_2022.dat', sep=';',
#           float_format='%.5f', na_rep='NaN', index=False)

# forcing
forc.to_csv(r'c:\data\GFB\FI-Var\proc\FI-Var_forcing_2013-2022.dat', sep=';', 
            na_rep='NaN', float_format='%.2f', index=False)

# forcing readme

Readme = 'Readme for FI-Var_forcing_2013-2022.dat'
Readme += "\n\nGap-Filled meteorological and soil data for model forcing (pyAPES)\n\nSamuli Launiainen, Luke" 
Readme += "\n\nYear, Month, Day, Hour, Minute: datetime [UTC + %.1f]" % timezone
Readme += "\n\nMissing data: NaN\n"
Readme += readme
outF = open(r'c:\data\GFB\FI-Var\proc\FI-Var_forcing_2013-2022_readme.txt', 'w')
print(Readme, file=outF)
outF.close()

# fmi observations
fmi.insert(0, 'Year', fmi.index.year.values)
fmi.insert(1, 'Month', fmi.index.month.values)
fmi.insert(2, 'Day', fmi.index.day.values)
fmi.insert(3, 'Hour', fmi.index.hour.values)
fmi.insert(4, 'Minute', fmi.index.minute.values)

fmi.to_csv(r'c:\DATA\GFB\FI-Var\Proc\FI-Var_fmiobs_2013_2022.dat', sep=';',
           float_format='%.5f', na_rep='NaN', index=False)


