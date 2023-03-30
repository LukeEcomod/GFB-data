# -*- coding: utf-8 -*-
"""
.. module: utils
    :synopsis: utils for reading, plotting and analyzing flux and meteorological timeseries data
.. moduleauthor:: Samuli Launiainen

Last edit 21.3.2023 / Samuli

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

#to retrieve FMI weather data
from fmiopendata.wfs import download_stored_query
import datetime
from pyproj import Geod
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn import preprocessing

EPS = np.finfo(float()).eps

#: [J mol\ :sup:`-1`\ ], latent heat of vaporization at 20\ :math:`^{\circ}`\ C
LATENT_HEAT = 44100.0
#: [kg mol\ :sup:`-1`\ ], molar mass of H\ :sub:`2`\ O
MOLAR_MASS_H2O = 18.015e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of CO\ :sub:`2`\
MOLAR_MASS_CO2 = 44.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of C
MOLAR_MASS_C = 12.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of air
MOLAR_MASS_AIR = 29.0e-3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of H\ :sub:`2`\ O
SPECIFIC_HEAT_H2O = 4.18e3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of organic matter
SPECIFIC_HEAT_ORGANIC_MATTER = 1.92e3
#: [J mol\ :sup:`-1` K\ :sup:`-1`\ ], heat capacity of air at constant pressure
SPECIFIC_HEAT_AIR = 29.3
#: [W m\ :sup:`-2` K\ :sup:`-4`\ ], Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.6697e-8
#: [-], von Karman constant
VON_KARMAN = 0.41
#: [K], zero degrees celsius in Kelvin
DEG_TO_KELVIN = 273.15
#: [K], zero degrees celsius in Kelvin
NORMAL_TEMPERATURE = 273.15
#: [mol m\ :sup:`-3`\ ], density of air at 20\ :math:`^{\circ}`\ C
AIR_DENSITY = 41.6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], kinematic viscosity of air at 20\ :math:`^{\circ}`\ C
AIR_VISCOSITY = 15.1e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], thermal diffusivity of air at 20\ :math:`^{\circ}`\ C
THERMAL_DIFFUSIVITY_AIR = 21.4e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of CO\ :sub:`2` at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_CO2 = 15.7e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of H\ :sub:`2`\ at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_H2O = 24.0e-6
#: [J mol\ :sup:`-1` K\ :sup:``-1], universal gas constant
GAS_CONSTANT = 8.314
#: [kg m\ :sup:`2` s\ :sup:`-1`\ ], standard gravity
GRAVITY = 9.81
#: [kg m\ :sup:`-3`\ ], water density
WATER_DENSITY = 1.0e3
#: [umol m\ :sup:`2` s\ :sup:`-1`\ ], conversion from watts to micromol
PAR_TO_UMOL = 4.56
#: [rad], conversion from deg to rad
DEG_TO_RAD = 3.14159 / 180.0
#: [umol m\ :sup:`-1`], O2 concentration in air
O2_IN_AIR = 2.10e5

def read_avaa_ascii(ffile, format='avaa-data', sep='\t', decimal='.', UTC=False):
    """
    Reads ascii-file in avaa-data .txt download format 
    Avaa-data is in UTC+2.
    Args:
        ffile (str) - filepath
        format (str) - 'avaa-data'
    Returns:
        data (pd.DataFrame) - all data merged to a single dataframe
    """
    
    #specs = {'avaa-data': {'sep': '\t', 'decimal': '.'}}
    
    data = pd.read_csv(ffile, sep='\t', decimal='.', header='infer', parse_dates=True, infer_datetime_format=True)
    t = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    
    data.index = t
    
    #convert to utc & nearest neighbour interpolation to last 4 periods
    #DatetimeIndex.tz_localize() would account for DST; the data in always in UTC+2
    if UTC:
        data = data.shift(periods=-4)
        data.iloc[-4:,:] = data.iloc[-5,:].values
        
    return data

def get_avaa_dataset(ffiles, UTC=False):
    """
    Returns dataset, combines timeseries from several files
    
    """
    dlist = []
    N = len(ffiles)
    
    for k in range(N):
        dlist.append(read_avaa_ascii(ffiles[k], UTC=UTC))
    
    # join dataframes, omit common columns
    df = dlist[0]
    for k in range(1,N):
        cols_to_use = dlist[k].columns.difference(df.columns)
        df = df.join(dlist[k][cols_to_use])
    
    return df

def RetrieveFMIObs(latin, lonin, ts, te, deltat, distlm=50000, resample=True,
                   varsin=None, dattype='weather'):
    """
    FROM Olli Peltola / PyECPC_utilities/tools.py
    
    Retrieves FMI open data from the closest FMI site.

    Args:
        latin (float): Latitude of the location in degrees (e.g. 63.13808441).
        lonin (float): Longitude of the location in degrees (e.g. 27.24616241).
        ts - start date ('yyyy-mm-dd')
        te - end date ('yyyy-mm-dd')
        deltat - frequency, e.g. '30min', '1D'
        distlm (int, optional): Upper limit for distance (m). Defaults to 50000.
        resample (bool): True if resample to 'freq'
        varsin (list, optional): Variables that are fetched. Defaults to None.
        site (str, optional): FMI site ID. Defaults to None.
        dattype (str, optional): Type of data to be fetched. Defaults to 'weather'.

   Returns:
       DataFrame: Retrieved data. DatetimeIndex, UTC+0.
       dict: units for the retrieved data.
       str: Site ID for the site from which data were retrieved.

    """
    
    time = pd.date_range(start=ts,end=te,freq=deltat).to_series()
    time.reset_index(drop=True,inplace=True)
    dates = time.dt.date.unique()

    # --- collect data in daily loop
    df = pd.DataFrame()
    
    print('fetching data...')
    for date in dates:
        print(date)
        ts = np.min(time[time.dt.date == date]).to_pydatetime()
        te = ts + datetime.timedelta(days=1)
        
        # Convert times to properly formatted strings
        ts = ts.isoformat(timespec="seconds") + "Z"
        te = te.isoformat(timespec="seconds") + "Z"

        datout, unitsout, site = RetrieveFMIObservations(latin,
                                                         lonin,
                                                         ts,
                                                         te,
                                                         distlm=distlm,
                                                         varsin=varsin,
                                                         dattype=dattype)

        #df = df.append(datout, ignore_index = True)
        df = pd.concat([df, datout])
    print('...done!')
    
    if 'time' in df.columns:        
        # set to datetime index, drop duplicates        
        df.sort_values('time', inplace=True)
        df.index = df['time']
        
        idx = np.unique(df.index.values, return_index = True )[1]
        df = df.iloc[idx]
        
        if resample:
            # interpoltate to match data frequency with 'deltat'
            dfout = pd.DataFrame(index = time)
            
            variables = list(df.columns)
            variables.remove('time')
            
            #if 'P_RAIN_FMI' in variables:
            #    variables.remove('P_RAIN_FMI')

            # time in float64
            tin = np.array(df['time']).astype('float64')
            tout = np.array(time).astype('float64')
            for vr in variables:
                # interpolate linearly
                linfit = interp1d(tin, df[vr],bounds_error=False)
                val = linfit(tout)
                dfout[vr] = val
                
                if vr=='P_RAIN_FMI': #accumulated precipitation
                    # retrieves nearest value
                    linfit = interp1d(tin, df[vr], bounds_error=False, kind='previous')
                    val = linfit(tout)
                    dfout[vr] = val
                    # no precip can be coded as -1.
                    dfout[vr][dfout[vr] < 0] = 0.0
                    dfout[vr].fillna(0.0, inplace=True)
                    
        else:
            dfout = df.copy()
            dfout.drop('time', axis=1, inplace=True)
    else:
        dfout = pd.DataFrame()
        dfout.index = time
        unitsout = dict()
        unitsout.update({'time':''})
        site = None
        
    return dfout,unitsout,site
    

def RetrieveFMIObservations(latin, lonin, start_time, end_time, distlm=50000, 
                            varsin=None, site=None, dattype='weather'):
    """
    Retrieves FMI open data from the closest FMI site.

    Args:
        latin (float): Latitude of the location in degrees (e.g. 63.13808441).
        lonin (float): Longitude of the location in degrees (e.g. 27.24616241).
        ts (str): start of the time period for which data is fetched, in format 2020-07-07T12:00:00Z
        te (str): End of the time period for which data is fetched.
        distlm (int, optional): Upper limit for distance (m). Defaults to 50000.
        varsin (list, optional): Variables that are fetched. Defaults to None.
        site (str, optional): FMI site ID. Defaults to None.
        dattype (str, optional): Type of data to be fetched. Defaults to 'weather'.

   Returns:
       DataFrame: Retrieved data.
       dict: units for the retrieved data.
       str: Site ID for the site from which data were retrieved.

    """
    FMIvm = {   'n_man':'',
                'p_sea':'PA_FMI',
                'r_1h':'P_RAIN_FMI',
                'rh':'RH_FMI',
                'ri_10min':'P_RAININTENSITY_FMI',
                'snow_aws':'D_SNOW_FMI',
                't2m':'TA_FMI',
                'td':'TD_FMI',
                'vis':'VISIBILITY_FMI',
                'wawa':'wawa_FMI',
                'wd_10min':'wind_dir_FMI',
                'wg_10min':'wind_gust_FMI',
                'ws_10min':'wind_speed_FMI',
                'LWIN_1MIN':'LWIN_FMI',
                'LWOUT_1MIN':'LWOUT_FMI',
                'GLOB_1MIN':'SWIN_FMI',
                'DIR_1MIN':'SWINDIR_FMI',
                'REFL_1MIN':'SWOUT_FMI',
                'SUND_1MIN':'SUND_FMI',
                'DIFF_1MIN':'SWINDIFF_FMI',
                'NET_1MIN':'RN_FMI',
                'UVB_U':'UVB_FMI'
            }
    
    # Distance to site will be measured on this ellipsoid - more accurate than a spherical method             
    wgs84_geod = Geod(ellps='WGS84') 


    # Convert times to properly formatted strings
    #start_time = ts.isoformat(timespec="seconds") + "Z"
    # -> 2020-07-07T12:00:00Z
    #end_time = te.isoformat(timespec="seconds") + "Z"
    # -> 2020-07-07T13:00:00Z
    
    # retrieve data
    if dattype == 'weather':
        
        obs = download_stored_query("fmi::observations::weather::multipointcoverage",
                                    args=['bbox=' + f"{lonin-1:.0f}" + ',' + f"{latin-1:.0f}" + ',' + f"{lonin+1:.0f}" + ',' + f"{latin+1:.0f}",
                                        "starttime=" + start_time,
                                        "endtime=" + end_time,"timeseries=True"])
    elif dattype == 'radiation':
        radsites = ['102035','101932','101920','107201','101787','101756','101586',
                    '101339','101104','100968','101030','101004','100908','100929']
        sitestr = ''
        for rsite in radsites:
            sitestr = sitestr + '&fmisid=' + rsite
            
        obs = download_stored_query("fmi::observations::radiation::multipointcoverage",
                            args=[sitestr,
                                  "starttime=" + start_time,
                                  "endtime=" + end_time,"timeseries=True"])

    if site is None:
        # FMI SYNOP station locations
        lats = list()
        lons = list()
        dists = list()
        for key in obs.data.keys():
            lat = obs.location_metadata[key]['latitude']
            lon = obs.location_metadata[key]['longitude']
            az12,az21,dist = wgs84_geod.inv(lonin,latin,lon,lat)

            lats.append(lat)
            lons.append(lon)
            dists.append(dist)

        # extracting data from the closest station
        FMIsites = list(obs.data.keys())
        if np.min(dists)<distlm:
            indx = [i for i, j in enumerate(dists) if j == np.min(dists)]
            indx = indx[0]
            site = FMIsites[indx]
        else:
            # no FMI site close to the EC site
            site = None

    if varsin is None:
        if site is not None:
            varsin = list(obs.data[site].keys())
            varsin.remove('times')


    datout = pd.DataFrame()
    unitsout = dict()
    if site is not None:
        for indx in range(len(varsin)):
            vr = varsin[indx]
            if vr!='times':
                if FMIvm[vr]=='D_SNOW_FMI':
                    # converting snow depth to m from cm to be relative to snow depth at EC sites
                    datout[FMIvm[vr]] = obs.data[site][vr]['values']
                    datout[FMIvm[vr]] = datout[FMIvm[vr]]*1e-2
                    unitsout.update({FMIvm[vr]:'m'})
                else:
                    datout[FMIvm[vr]] = obs.data[site][vr]['values']
                    # datout[indx] = obs.data[site][vr]['values']
                    unitsout.update({FMIvm[vr]:obs.data[site][vr]['unit']})

        datout['time'] = obs.data[site]['times']
        unitsout.update({'time':''})

    return datout, unitsout, site

   
def rf_crossvalidate(X, Y, test_size=0.3, nfolds=5, n_estimators=10):
    """
    Fits Random Forest model to data and performs nfolds cross-validation
    Args:
        X (N x marray) - features
        Y (N x1 array) - dependent variable
        test_size (-) - fraction of test data
        nfolds (-) - N-fold cross-validation
        n_estimators (-) - nr. estimators, see sklearn tutorial
    Returns:
        rf - random forest instance
        scaler - StandardScaler instance
        cv_score - cross-validation score (R2) mean
        score - test score (R2)
    """
    #from sklearn.ensemble import RandomForestRegressor
    #from sklearn.model_selection import cross_val_score, train_test_split
    #from sklearn import metrics
    #from sklearn import preprocessing
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    
    scaler = preprocessing.StandardScaler().fit(X)
    X_transformed = scaler.transform(X)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    
    # cross-validation

    rf = RandomForestRegressor(n_estimators=n_estimators).fit(X_transformed, Y)
    cv_score = cross_val_score(rf, X, Y, cv=nfolds)
    print('cross-validation score r2', cv_score)
    
    rf = RandomForestRegressor(n_estimators=n_estimators).fit(X_train_transformed, Y_train)
    score = rf.score(X_test_transformed, Y_test)
    print('test score r2', score)
    
    return rf, scaler, np.mean(cv_score), score
    

#%% Biomet gap-filling & pyAPES forcing file
def gap_fill_rf(Y, X, ix_t, n_estimators=10, figtitle=''):
    """
    Gap-filling of meteorological data using RandomForestRegressor
    Args:
        Y (Nx1 array) - dependent variable
        X (Nxm array) - features
        ix_t (array) - indices for test data in Y
    Returns:
        Y (array) - gap-filled Y
        Ypred (array) - predicted Y
        
    """
    t = np.arange(0, len(Y))
    ix_p = np.setdiff1d(t, ix_t)

    # fit random forest model
    rf, scaler, cv_score, score = rf_crossvalidate(X[ix_t], Y[ix_t], test_size=0.3, nfolds=5, n_estimators=10)
    Y_pred = rf.predict(scaler.transform(X))
    
    fig, ax = plt.subplots(2,2)
    
    ax[0,0].plot(t, Y_pred, '.-', alpha=0.5, label='RF')
    ax[0,0].plot(t[ix_t], Y[ix_t], '.-', alpha=0.5, label='train')
    ax[0,0].legend()
    ax[0,0].set_title(figtitle)
 

    #--- this computes histograms and plots
    def plothist(ax, y, bins=20, density=True, label='', alpha=0.5):
        h, bin_edges = np.histogram(y, bins=bins, density=density)
        x = bin_edges[:-1]
        w = np.diff(bin_edges)
 
        ax.bar(x, h, width=w, align='edge', alpha=alpha, label=label)
        
    
    plothist(ax[1,0], Y[ix_t], bins=20, alpha = 0.5, density=True, label='train')
    plothist(ax[1,0], Y_pred[ix_t], bins=20, alpha = 0.5, density=True, label='RF - train')
    plothist(ax[1,0], Y_pred[ix_p], bins=20, alpha = 0.5, density=True, label='RF - pred')
    ax[1,0].legend()
    
    ax[1,1].plot(Y[ix_t], Y_pred[ix_t], '.', alpha=0.3)
    p = np.polyfit(Y[ix_t], Y_pred[ix_t], 1)
    xx = np.array([min(Y[ix_t]), max(Y[ix_t])])
    fit = p[1] + p[0] * xx
    ax[1,1].plot(xx, fit, '-', label='%.2f' %p[0])
    ax[1,1].legend()
    
    Y[ix_p] = Y_pred[ix_p]

    return Y, Y_pred

def InterpolateShortGaps(dat, N, method='linear', flags=None, gapcode=1):
    """Filling short gaps within the time series with interpolation.
    Args:
        df (pd.Series): data
        N (int): max length (observations) for the gaps to be filled
        method (str, optional): method used for interpolation. Default 'linear'
        flags (pd.Series) - optional
        gapcode (int) - code for short gaps
    Returns:
        out (pd.Series): output data, the short gaps filled with selected interpolation method
        flag (pd.Series): gap-flags. 0 = observations, gapcode = short gaps
    """
    
    out = dat.copy()
    
    if (method!='linear') & (method!='nearest') & (method!='quadratic') & (method!='cubic') & (method!='spline'):
        msg = 'Unknown method given (%s) for interpolating short gaps. Using ''linear'' instead.'%(method,)
        print(msg)
        method = 'linear'
        
    dat_interpolated = dat.interpolate(method=method)

    mask = dat.isna()
    x = (mask.groupby((mask != mask.shift()).cumsum()).transform(lambda x: len(x) > N)*mask)
    
    ixn = np.where(~x)[0] # indices that are either observations or match short gap criteria
    out[ixn] = dat_interpolated[ixn] # ... values copied to output
    del ixn
    
    if flags is None:
        flags = pd.Series(data=np.ones(len(dat))*np.NaN, index=dat.index)
        # observations
        ixg = np.where(dat.isna()==False)[0]
        flags.iloc[ixg] = 0
    
    # get filled indices
    diff = dat.reset_index().compare(out.reset_index())

    ix = diff.index
    flags.iloc[ix] = gapcode
    
    return out, flags

def MDC(dat, method='mean', deltat='30min', flags=None):
    """
    Gapfilling by mean diurnal course
    dat, method='mean', deltadays=1, Nmin=1, deltat='30min'
    
    Args:
        dat (pd.Series) - data
        method (str) - 'mean', 'median'
    Returns:
        dout (pd.Series) - gap-filled series
        flags (pd.Series) - flags for window length
    """
    from src.utils import DiurnalPattern
    
    Nmin = 2
    dout = dat.copy()
    N = len(dat)
    
    if flags is None:
        flags = pd.Series(dat=np.ones(N)*np.NaN, index=dat.index)
        # observations
        ixg = np.where(dat.isna()==False)[0]
        flags.iloc[ixg] = 0

    gaps = True    
    window = 2
    
    while gaps:
        #print(window)
        fdat = DiurnalPattern(dout, method=method, deltadays=window, Nmin=Nmin, deltat=deltat)
        ix = np.where(dout.isna())
        #ix = np.where((dout.isna()==True) & (fdat.isna()==False))[0]
        
        dout.iloc[ix] = fdat.iloc[ix]
        flags.iloc[ix] = window 
        
        window += 2
        
        # bool
        gaps = dout.isna().any()
        #print(len(np.where(dout.isna())[0]))
    
    return dout, flags

def DiurnalPattern(dat, method='mean', deltadays=1, Nmin=1, deltat='30min'):
    """
    Args:
        dat (pd.Series)
        method (str)
        deltadays (int) - period length (days)
        Nmin (int) - data from 
        deltat (str) - '30min'
    
    Returns:
        pd.Series - signal reconstructed by mean diurnal averaging in deltadays window
    """
    
    #Nmin = min(Nmin, deltadays)
    Nmin = max(2, int(np.ceil(deltadays/2)))

    if (method!='mean') and (method!='median') and (method!='std'):
        msg = 'Unknown method given (%s) for calculating diurnal patterns. Using ''mean'' instead.'%(method,)
        print(msg)
        method = 'mean'

    time = dat.index.to_pydatetime()
    tstart = time[0].date()
    tend = time[-1].date()
    t = pd.date_range(start=tstart, end=tend + datetime.timedelta(days=1), freq=deltat)
    t = t[0:-1]
    dat = dat.reindex(t)
    
    t = pd.Series(t,index=t)

    date = (t - t[0]) /datetime.timedelta(days=1)
    time2 = ((date-np.floor(date))*24)
    time2 = time2.round(decimals=5)
    time2 = time2.drop_duplicates().to_numpy()
    date = np.floor(date).unique()

    z3 = pd.DataFrame(data=dat.to_numpy().reshape(len(date),len(time2)))
    
    if method=='mean':
        z4 = pd.Series(data=z3.rolling(deltadays,center=True,min_periods=Nmin).mean().to_numpy().reshape(len(t),).T,index=t)
    elif method=='median':
        z4 = pd.Series(data=z3.rolling(deltadays,center=True,min_periods=Nmin).median().to_numpy().reshape(len(t),).T,index=t)
    elif method=='std':
        z4 = pd.Series(data=z3.rolling(deltadays,center=True,min_periods=Nmin).std().to_numpy().reshape(len(t),).T,index=t)

    datout = z4.reindex(index=dat.index)

    return datout

#%% pyAPES forcing file generation

def create_forcingfile(df, output_file, lat, lon, met_data=None, timezone=+0.0, fpar=0.45,
                       CO2_constant=None, short_gap_len=5):
    """
    Create forcing file from meteorological observations
    Args:
        df (dataframe)
        output_file (str): name of output file (.csv not included)
        lat (float): latitude, decimal format
        lon (float): longitude
        P_unit (float): unit conversion needed to get to [Pa]
        met_data (dataframe) - meteorological observations from nearest weather station
    """
    tau = 8.33 # S-model parameter, Kolari et al. 2007 Tellus B
    #tau = 7.0 # Kolari et al. 2014 Front. Plant. Sci
    P_unit = 1.0
    N = len(df) # nobs
    
    # store data flags: 0 - observed
    flagcols = ['Tair', 'RH', 'H2O', 'U', 'u_star', 'P', 'CO2', 'Prec', 
                'Rg', 'Par', 'LWin', 'Tsoil', 'Wsoil', 'Snowdepth']
    
    flags = pd.DataFrame(data=np.zeros(shape=(N, len(flagcols))), 
                         columns=flagcols)
    
    # timestep
    dt = (df.index[1] - df.index[0]).total_seconds()
    dfcols = df.columns

    cols = []
    readme = ''

    # day of year
    df['doy'] = df.index.dayofyear
    cols.append('doy')
    readme += "\ndoy: Day of year [days]"

    # precipitation
    cols.append('Prec')
    readme += "\nPrec: Precipitation [mm/30min]"

    # atm. pressure unit from [XPa] to [Pa]
    df['P'] = df['P'] * P_unit
    cols.append('P')
    readme += "\nP: Ambient pressure [Pa]"

    # air temperature: instant and daily [degC]
    cols.append('Tair')
    readme += "\nTair: Air temperature [degC]"

    # wind speed [ms-1]
    cols.append('U')
    readme += "\nU: Wind speed [m/s]"
    # friction velocity [ms-1]
    cols.append('u_star')
    readme += "\nu_star: Friction velocity [m/s]"

    # ambient H2O [mmol/mol] from RH
    esat, _ = e_sat(df['Tair'])
    df['H2O'] = 1e3 * (df['RH'] / 100.0) * esat / df['P']
    cols.append('H2O')
    readme += "\nH2O: Ambient H2O [mmol/mol]"
    
    if met_data is not None:
        esat, _ = e_sat(met_data['Tair'])
        met_data['H2O'] = 1e3* (met_data['RH'] / 100.0) * esat / 101300.0
        
    # ambient CO2 [ppm]
    readme += "\nCO2: Ambient CO2 [ppm]"
    if 'CO2' not in df or CO2_constant:
        df['CO2'] = CO2_constant
        readme += " - set constant!"
    cols.append('CO2')

    # --- gap-fill Tair, RH, U, ustar, Prec, P, CO2
    # 1) gaps shorter than short_gap_len by linear interpolation, flag = 1
    # 2) By nearby met-station data (if exists), flag = -1
    # 3) By MDC in 2, 4, 6, ... day windows, flag = window
    
    fillcols = ['Tair', 'RH', 'U', 'u_star', 'P', 'CO2', 'RH', 'H2O']

    print('*** filling gaps in: ')
    for c in fillcols:
        print(c)
        # interpolate short gaps using linear interpolation
        flags[c].iloc[df[c].isna()==False] = 0 # observed
        df[c], flags[c] = InterpolateShortGaps(df[c], N=short_gap_len, method='linear', 
                                                flags=flags[c],gapcode=1)
        
        # if variable in meteostation data, use that
        if met_data is not None:
            if c in met_data.columns:
                xx = np.where(np.isnan(df[c]))[0]
                xxm = np.where(np.isfinite(met_data[c]))[0]
                ix = np.nonzero(np.in1d(xxm, xx))[0]
                
                df[c].iloc[ix] = met_data[c].iloc[ix].values
                flags[c].iloc[ix] = -1
        
        # apply MDV to fill remindig gaps
        df[c], flags[c] = MDC(df[c], method='median', deltat='30min', flags=flags[c]) 
        
    # *** precipitation (mm/30min)
    c = 'Prec'
    ix = np.where(df[c].isna()==False)[0]
    ixg = np.where(df[c].isna())[0]
    flags[c].iloc[ix] = 0 # observed
    flags[c].iloc[ixg] = -2 # potentially missing, set to 0.0
    
    # if variable in meteostation data, use that
    if met_data is not None:
        if c in met_data.columns:
            xx = np.where(np.isnan(df[c]))[0]
            xxm = np.where(np.isfinite(met_data[c]))[0]
            ix = np.nonzero(np.in1d(xxm, xx))[0]
            
            df[c].iloc[ix] = met_data[c].iloc[ix].values
            flags[c].iloc[ix] = -1
    else:
        df['Prec'] = df['Prec'].fillna(0.0)
        
    # *** solar zenith angle
    jday = df.index.dayofyear + df.index.hour / 24.0 + df.index.minute / 1440.0 - dt / 2.0 / 86400.0

    df['Zen'], _, _, _, _, _ = solar_angles(lat, lon, jday, timezone=timezone)
    cols.append('Zen')
    readme += "\nZen: Zenith angle [rad], (lat = %.2f, lon = %.2f)" % (lat, lon)

    # *** Estimate and gap-fill radiation components
    
    # Par and Rg are dependent so replace gaps in each other; proportionality is summertime median ratio
    
    ix = ((df.index.month >= 5) & (df.index.month <= 9))
    fpar = np.nanmedian(df['Rg'][ix] / (df['Par'][ix] + EPS))
    del ix
    
    ix = np.where(np.isnan(df['Rg']))[0]
    flags['Rg'].iloc[ix] = -1
    df['Rg'].iloc[ix] = df['Par'].iloc[ix] * fpar # Wm-2 
    print('Rg gaps', len(ix))
    del ix

    ix = np.where(np.isnan(df['Par']))[0]
    flags['Par'].iloc[ix] = -1
    df['Par'].iloc[ix] = df['Rg'].iloc[ix] / fpar # umol m-2
    print('Par gaps', len(ix))
    del ix
    
    print('*** filling gaps in: ')
    for c in ['Rg', 'Par']:
        print(c)
        # interpolate short gaps using linear interpolation
        flags[c].iloc[df[c].isna()==False] = 0 # observed
        df[c], flags[c] = InterpolateShortGaps(df[c], N=short_gap_len, method='linear', 
                                                flags=flags[c],gapcode=1)
        
        # if variable in meteostation data, use that
        if met_data is not None:
            if c in met_data.columns:
                xx = np.where(np.isnan(df[c]))[0]
                xxm = np.where(np.isfinite(met_data[c]))[0]
                ix = np.nonzero(np.in1d(xxm, xx))[0]
                
                df[c].iloc[ix] = met_data[c].iloc[ix].values
                flags[c].iloc[ix] = -1
        
        # apply MDV to fill remainding gaps
        df[c], flags[c] = MDC(df[c], method='median', deltat='30min', flags=flags[c]) 
    
    plt.figure()
    plt.plot(df['Par'], '-')
    plt.plot(df['Rg'], '-')
    
    # --- estimate diffuse fraction
    
    f_diff = pd.Series(np.ones(N)*np.NaN, index=df.index)
    
    # measured diffuse fraction
    if 'diffRg' in dfcols:
        ix = np.where(df['Rg']>50.0)[0]
        f_diff.iloc[ix] = df['diffRg'].iloc[ix] / (df['Rg'].iloc[ix] + EPS) # based on Rg
    elif 'diffPar' in dfcols:
        ix = np.where((f_diff.isna()) & (df['Par']> 100.0))[0]
        f_diff.iloc[ix] = df['diffPar'].iloc[ix] / (df['Par'].iloc[ix] + EPS) # based on Par
        
    f_diff[f_diff<0.2] = 0.2
    f_diff[f_diff>1.0] = 1.0
    # when sun below horizon or Par < 10 umolm-2, assume all is diffuse.
    f_diff[df['Zen'] <= 0] = 1.0
    f_diff[df['Par'] < 10] = 1.0

    # gap-fill with Random Forest
    ix_t = np.where(f_diff.isna() == False)[0]

    X = df[['doy', 'Zen', 'Rg']]
    X['doy'] = X['doy'] + X.index.hour.values / 24 + X.index.hour.values / (24 * 60)

    X = X.values
    Y = f_diff.values
    
    ygf, _ = gap_fill_rf(Y, X, ix_t, n_estimators=10, figtitle='d_f')
    
    f_diff = pd.Series(ygf, index=df.index)
    
    # estimate short-wave components
    df['diffPar'] = f_diff * fpar * df['Rg']
    df['dirPar'] = (1 - f_diff) * fpar * df['Rg']
    df['diffNir'] = f_diff * (1 - fpar) * df['Rg']
    df['dirNir'] = (1 - f_diff) * (1 - fpar) * df['Rg']

    cols.extend(('diffPar', 'dirPar', 'diffNir', 'dirNir'))
    readme += "\ndiffPar: Diffuse PAR [W/m2], estimated from Rg and diffuse fraction \ndirPar: Direct PAR [W/m2]"
    readme += "\ndiffNir: Diffuse NIR [W/m2] \ndirNir: Direct NIR [W/m2]"
    
    # estimate LW
    
    if 'LWin' in dfcols:
        
        #LWin = df['LWin']
        emi_sky = df['LWin'] / (STEFAN_BOLTZMANN * (df['Tair'] + DEG_TO_KELVIN)**4)
        emi_sky[emi_sky > 1.0] = 1.0

        ix_t = np.where(emi_sky.isna()==False)[0]
        
        # gap-fill with Random Forest
        Y = emi_sky.values

        X = df[['doy', 'Zen', 'Tair', 'H2O', 'Rg']]
        X['doy'] = X['doy'] + X.index.hour.values / 24 + X.index.hour.values / (24 * 60)

        X = X.values
        Y = f_diff.values
    
        emi_sky, _ = gap_fill_rf(Y, X, ix_t, n_estimators=10, figtitle='emi sky')
    
        
        df['LWin'] = emi_sky * STEFAN_BOLTZMANN * (df['Tair'] + DEG_TO_KELVIN)**4
        cols.append('LWin')
        readme += "\nLWin: Downward thermal radiation [W/m2], gap-filled with RF on: doy, H2O, Tair, Rg."
        
        flags['LWin'] = np.ones(N)
        flags['LWin'].iloc[ix_t] = 1.0
        
        # estimate clould cover and clear-sky emissivity
        decimal_doy = df['doy'].values + df.index.hour.values / 24 + df.index.minute.values / (24*60)
    
        f_cloud, _ , emi_sky = compute_clouds_rad(decimal_doy,
                                                  df['Zen'].values,
                                                  df['Rg'].values,
                                                  1e-3 * df['H2O'].values * df['P'].values,
                                                  df['Tair'].values)
    
    
        emi_sky = pd.Series(emi_sky, index=df.index)
        emi_sky = emi_sky.interpolate('linear')
    
        df['LWin_mod'] = emi_sky * STEFAN_BOLTZMANN * (df['Tair'] + DEG_TO_KELVIN)**4
        cols.append('LWin_mod')
        readme += "\nLWin_mod: Downward thermal radiation [W/m2], modeled based on estimated cloud-cover and near-surface H2O and Tair"
        del emi_sky
    
    else:
        # estimate clould cover and clear-sky emissivity
        decimal_doy = df['doy'].values + df.index.hour.values / 24 + df.index.minute.values / (24*60)
    
        f_cloud, _ , emi_sky = compute_clouds_rad(decimal_doy,
                                                  df['Zen'].values,
                                                  df['Rg'].values,
                                                  1e-3 * df['H2O'].values * df['P'].values,
                                                  df['Tair'].values)
    
    
        emi_sky = pd.Series(emi_sky, index=df.index)
        emi_sky = emi_sky.interpolate('linear')
    
        df['LWin'] = emi_sky * STEFAN_BOLTZMANN * (df['Tair'] + DEG_TO_KELVIN)**4
        cols.append('LWin')
        readme += "\nLWin: Downward thermal radiation [W/m2], modeled based on estimated cloud-cover +\
                    and near-surface H2O and Tair"
    
    
    if {'Tsoil', 'Wliq'}.issubset(df.columns):
        cols.extend(('Tsoil', 'Wliq'))
        readme += "\nTsoil: Soil surface layer temperature [degC]]"
        readme += "\nWliq: Soil surface layer moisture content [m3 m-3]"
        
        print('*** filling gaps in: ')
        for c in ['Tsoil', 'Wsoil']:
            print(c)
            # interpolate short gaps using linear interpolation
            flags[c].iloc[df[c].isna()==False] = 0 # observed
            df[c], flags[c] = InterpolateShortGaps(df[c], N=short_gap_len, method='linear', 
                                                    flags=flags[c],gapcode=1)
            
            # apply MDV to fill remaining gaps
            df[c], flags[c] = MDC(df[c], method='median', deltat='30min', flags=flags[c])
            

    # daily mean temperature
    df['Tdaily'] = df['Tair'].resample('D').mean()
    df['Tdaily'] = df['Tdaily'].fillna(method='ffill')
    df['Tdaily'] = df['Tdaily'].fillna(method='bfill')
    
    cols.append('Tdaily')
    readme += "\nTdaily: Daily air temperature [degC]"

    X = np.zeros(N)
    DDsum = np.zeros(N)
    for k in range(1,N):
        if df['doy'][k] != df['doy'][k-1]:
            X[k] = X[k - 1] + 1.0 / tau * (df['Tdaily'][k-1] - X[k - 1])
            if df['doy'][k] == 1:  # reset in the beginning of the year
                DDsum[k] = 0.
            else:
                DDsum[k] = DDsum[k - 1] + max(0.0, df['Tdaily'][k-1] - 5.0)
        else:
            X[k] = X[k - 1]
            DDsum[k] = DDsum[k - 1]
    df['X'] = X
    cols.append('X')
    readme += "\nX: phenomodel delayed temperature [degC]"
    df['DDsum'] = DDsum
    cols.append('DDsum')
    readme += "\nDDsum: degreedays [days]"

    # Snowdepth

    if 'Snowdepth' in df.columns:
        c = 'Snowdepth'
        # interpolate short gaps using linear interpolation
        flags[c].iloc[df[c].isna()==False] = 0 # observed
        df[c], flags[c] = InterpolateShortGaps(df[c], N=10*48, method='linear', 
                                                flags=flags[c],gapcode=1)
        
        # if variable in meteostation data, use that
        if met_data is not None:
            if c in met_data.columns:
                xx = np.where(np.isnan(df[c]))[0]
                xxm = np.where(np.isfinite(met_data[c]))[0]
                ix = np.nonzero(np.in1d(xxm, xx))[0]
                
                df[c].iloc[ix] = met_data[c].iloc[ix].values
                flags[c].iloc[ix] = -1
        
        #df[c].fillna(method='ffill')
        #df[c].fillna(method='bfill')
        
        cols.append('Snowdepth')
        readme += "\nSnowdepth [m], not extrapolated outside measurements"
        
    # Checking timestamp validity
    # clear sky Global radiation at surface
    So = 1367
    df['Qclear'] = np.maximum(0.0,
            (So * (1.0 + 0.033 * np.cos(2.0 * np.pi * (np.minimum(df['doy'].values, 365) - 10) / 365)) * np.cos(df['Zen'].values)))
    df[['Qclear','Rg']].plot(kind='line')    
    
    forc = df[cols]
    #forc.plot(subplots=True, kind='line')

    print("/nNaN values in forcing data:")
    print(forc.isnull().any())
    
    return forc, flags, readme


def solar_angles(lat, lon, jday, timezone=+2.0):
    """
    computes zenith, azimuth and declination angles for given location and time
    Args:
        lat, lon (deg)
        jday - decimal day of year (float or array)
        timezone - > 0 when east from Greenwich
    Returns:
        zen, azim, decl - rad
        sunrise, sunset, daylength (minutes)
    Algorithm based on NOAA solar calculator: https://www.esrl.noaa.gov/gmd/grad/solcalc/
    Equations: https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF
    """
    lat0 = lat * DEG_TO_RAD
    jday = np.array(jday, ndmin=1)

    # fract. year (rad)
    if np.max(jday) > 366:
        y = 2*np.pi / 366.0 * (jday - 1.0)
    else:
        y = 2*np.pi / 365.0 * (jday - 1.0)

    # declination angle (rad)
    decl = (6.918e-3 - 0.399912*np.cos(y) + 7.0257e-2*np.sin(y) - 6.758e-3*np.cos(2.*y)
        + 9.07e-4*np.sin(2.*y) - 2.697e-3*np.cos(3.*y) + 1.48e-3*np.sin(3.*y))

    # equation of time (min)
    et = 229.18*(7.5e-5 + 1.868e-3*np.cos(y) - 3.2077e-2*np.sin(y)
        - 1.4615e-2*np.cos(2.*y) - 4.0849e-2*np.sin(2.*y))
    # print et / 60.
    # hour angle
    offset = et + 4.*lon - 60.*timezone
    fday = np.modf(jday)[0]  # decimal day
    ha = DEG_TO_RAD * ((1440.0*fday + offset) / 4. - 180.)  # rad

    # zenith angle (rad)
    aa = np.sin(lat0)*np.sin(decl) + np.cos(lat0)*np.cos(decl)*np.cos(ha)
    zen = np.arccos(aa)
    del aa

    # azimuth angle, clockwise from north in rad
    aa = -(np.sin(decl) - np.sin(lat0)*np.cos(zen)) / (np.cos(lat0)*np.sin(zen))
    azim = np.arccos(aa)

    # sunrise, sunset, daylength
    zen0 = 90.833 * DEG_TO_RAD  # zenith angle at sunries/sunset after refraction correction

    aa = np.cos(zen0) / (np.cos(lat0)*np.cos(decl)) - np.tan(lat0)*np.tan(decl)
    ha0 = np.arccos(aa) / DEG_TO_RAD

    sunrise = 720.0 - 4.*(lon + ha0) - et  # minutes
    sunset = 720.0 - 4.*(lon - ha0) - et  # minutes

    daylength = (sunset - sunrise)  # minutes

    sunrise = sunrise + timezone
    sunrise[sunrise < 0] = sunrise[sunrise < 0] + 1440.0

    sunset = sunset + timezone
    sunset[sunset > 1440] = sunset[sunset > 1440] - 1440.0

    return zen, azim, decl, sunrise, sunset, daylength

def compute_clouds_rad(doy, zen, Rg, H2O, Tair):
    """
    Estimates atmospheric transmissivity (tau_atm [-]), cloud cover fraction
    (f_cloud (0-1), [-]) and fraction of diffuse to total SW radiation (f_diff, [-])
    Args:
        doy - julian day
        zen - sun zenith angle (rad)
        Rg - total measured Global radiation above canopy (Wm-2)
        H2O - water vapor pressure (Pa)
    Returns:
        f_cloud - cloud cover fraction (0-1) [-]
        f_diff - fraction of diffuse to total radiation (0-1), [-]
        emi_sky - atmospheric emissivity (0-1) [-]

    ! values for Rg < 100 W/m2 linearly interpolated

    Cloudiness estimate is based on Song et al. 2009 JGR 114, 2009, Appendix A & C
    Clear-sky emissivity as in Niemelä et al. 2001 Atm. Res 58: 1-18.
    eq. 18 and cloud correction as Maykut & Church 1973.
    Reasonable fit against Hyytiälä data (tested 20.6.13)

    Samuli Launiainen, METLA, 2011-2013
    """

    # solar constant at top of atm.
    So = 1367.0
    # clear sky Global radiation at surface
    Qclear = np.maximum(0.0,
                        (So * (1.0 + 0.033 * np.cos(2.0 * np.pi * (np.minimum(doy, 365) - 10) / 365)) * np.cos(zen)))
    tau_atm = Rg / (Qclear + EPS)

    # cloud cover fraction
    f_cloud = np.maximum(0, 1.0 - tau_atm / 0.7)

    # calculate fraction of diffuse to total Global radiation: Song et al. 2009 JGR eq. A17.
    f_diff = np.ones(f_cloud.shape)
    f_diff[tau_atm > 0.7] = 0.2

    ind = np.where((tau_atm >= 0.3) & (tau_atm <= 0.7))
    f_diff[ind] = 1.0 - 2.0 * (tau_atm[ind] - 0.3)

    # clear-sky atmospheric emissivity
    ea = H2O / 100  # near-surface vapor pressure (hPa)
#    emi0 = np.where(ea >= 2.0, 0.72 + 0.009 * (ea - 2.0), 0.72 -0.076 * (ea - 2.0))
    emi0 = 1.24 * (ea/(Tair + 273.15))**(1./7.) # Song et al 2009

    # all-sky emissivity (cloud-corrections)
#    emi_sky = (1.0 + 0.22 * f_cloud**2.75) * emi0  # Maykut & Church (1973)
    emi_sky = (1 - 0.84 * f_cloud) * emi0 + 0.84 * f_cloud  # Song et al 2009 / (Unsworth & Monteith, 1975)

#    other emissivity formulas tested
#    emi_sky=(1 + 0.2*f_cloud)*emi0;
#    emi_sky=(1 + 0.3*f_cloud.^2.75)*emi0; % Maykut & Church (1973)
#    emi_sky=(1 + (1./emi0 -1)*0.87.*f_cloud^3.49).*emi0; % Niemelä et al. 2001 eq. 15 assuming Ts = Ta and surface emissivity = 1
#    emi_sky=(1-0.84*f_cloud)*emi0 + 0.84*f_cloud; % atmospheric emissivity (Unsworth & Monteith, 1975)

#    f_cloud[Rg < 100] = np.nan
#    f_diff[Rg < 100] = np.nan
#    emi_sky[Rg < 100] = np.nan

    f_cloud[Qclear < 10] = np.nan
    f_diff[Qclear < 10] = np.nan
    emi_sky[Qclear < 10] = np.nan

    df = pd.DataFrame({'f_cloud': f_cloud, 'f_diff': f_diff, 'emi_sky': emi_sky})
    df = df.interpolate()
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')

    return df['f_cloud'].values, df['f_diff'].values, df['emi_sky'].values

def e_sat(T):
    """
    Computes saturation vapor pressure (Pa), slope of vapor pressure curve
    [Pa K-1]  and psychrometric constant [Pa K-1]
    IN:
        T - air temperature (degC)
    OUT:
        esa - saturation vapor pressure in Pa
        s - slope of saturation vapor pressure curve (Pa K-1)
    SOURCE:
        Campbell & Norman, 1998. Introduction to Environmental Biophysics. (p.41)
    """

    esa = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
    s = 17.502 * 240.97 * esa / ((240.97 + T)**2)

    return esa, s

def latent_heat(T):
    """
    Computes latent heat of vaporization or sublimation [J/kg]
    Args:
        T: ambient air temperature [degC]
    Returns:
        L: latent heat of vaporization or sublimation depending on 'type'[J/kg]
    """
    # latent heat of vaporizati [J/kg]
    Lv = 1e3 * (3147.5 - 2.37 * (T + DEG_TO_KELVIN))
    # latent heat sublimation [J/kg]
    Ls = Lv + 3.3e5
    L = np.where(T < 0, Ls, Lv)
    return L

def save_df_to_csv(df, fn, readme='', timezone=+2, sep=';'):
    """
    Save dataframe with datetime index to csv file with corresponding readme.txt.
    Args:
        df (DataFrame): data to save
        fn (str): filename of saved file (.dat not included)
        readme (str): readme corresponding to df to save as txt file
        fp (str): filepath, forcing folder used as default
        timezone (float): time zone in refernce to UTC, default UTC + 2
    """

    # add datetime as columns
    df.insert(0, 'year', df.index.year.values)
    df.insert(1, 'month', df.index.month.values)
    df.insert(2, 'day', df.index.day.values)
    df.insert(3, 'hour', df.index.hour.values)
    df.insert(4, 'minute', df.index.minute.values)

    df.to_csv(path_or_buf= fn + ".dat", sep=sep, na_rep='NaN', index=False)
    Readme = "Readme for " + fn + ".dat"
    Readme += "\n\nSamuli Launiainen, Luke " + str(datetime.datetime.now().date())
    Readme += "\n\nyyyy, mo, dd, hh, mm: datetime [UTC + %.1f]" % timezone
    Readme += readme
    outF = open(fn + "_readme.txt", "w")
    print(Readme, file=outF)
    outF.close()
