'''
Author:  OPeltola (olli.peltola@luke.fi)
-----
File Created: Wednesday October 6th 2021
Last Modified: Wednesday, 6th October 2021 3:16:22 pm
Modified By:  OPeltola (olli.peltola@luke.fi)
-----
'''
from cmath import nan
import os, glob, shutil
import PyECPC_utilities.constants as c
import datetime as dt
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scipy.stats as stats
import scipy.odr as odr
import matplotlib.pyplot as plt
import numpy as np
import pysolar
import logging
from zipfile import ZipFile
from fmiopendata.wfs import download_stored_query
from pyproj import Geod
from PyECPC_utilities.ecpcvars import varsFMIname as FMIvm
from sklearn import linear_model, metrics

logger = logging.getLogger("ecpc_log")

def CircularStd(angles):

    s = np.sin(angles*np.pi/180.0)
    sa = s.mean()
    c = np.cos(angles*np.pi/180.0)
    ca = c.mean()

    # see Eq. (25) here https://doi.org/10.1175/1520-0450(1984)023<1362:ACOSPE>2.0.CO;2
    eps = (1-(ca**2+sa**2))**0.5
    std_angl = np.arcsin(eps)*(1+2/3**0.5*eps**3)

    return std_angl


def CircularMean(angles):

    s = np.sin(angles*np.pi/180.0)
    sa = s.mean()
    c = np.cos(angles*np.pi/180.0)
    ca = c.mean()

    
    # see Eq. (3) here https://doi.org/10.1175/1520-0450(1984)023<1362:ACOSPE>2.0.CO;2
    mean_angl = np.arctan(sa/ca)*180/np.pi+180

    # wrap within 0 - 360
    if mean_angl>360:mean_angl = mean_angl - 360
    if mean_angl<0:mean_angl = mean_angl + 360
    
    return mean_angl

def WindDirection(u,v,offset):

    WD = np.pi - np.arctan2(v,u)
    # WD = np.arctan2(u,v)
    # WD = np.arctan2(v,u)+np.pi
    # conversion to degrees
    WD = WD*180.0/np.pi

    # offset
    WD = WD + offset

    # wrap within 0 - 360
    WD[WD>360] = WD[WD>360]-360
    WD[WD<0] = WD[WD<0]+360
    
    return WD

def Fit_CF(U,fco,deg_wT,ref_wT,weights=None):

    if weights is None:
        weights = np.ones(np.shape(U))
        
        


    X = []
    for i in range(len(U)):
        X.append([U[i],fco[i]])
    X = np.array(X)
    y = np.array(ref_wT)/np.array(deg_wT)

    def mod(X,c1,c2):
        return c1*X[:,0]/(c2+X[:,1])+1

    fit = curve_fit(mod, X, y,sigma=weights)

    
    fitcoef = fit[0]
    fitcoef_sd = np.sqrt(np.diag(fit[1]))

    yfit = mod(X,fitcoef[0],fitcoef[1])

    # output
    MSE = np.nanmean((y-yfit)**2)
    r2 = 1-np.nansum((y-yfit)**2)/np.nansum((np.nanmean(y)-y)**2)
    res = {'coef':fitcoef,'coef_sd':fitcoef_sd,'fit':yfit,'meas':y,'r2':r2,'MSE':MSE}

    return res




def Fit_plane(u,v,w):
    """Fits the following equation to u,v,w data: w = bo + b1*u + b2*v. Used in deriving planar fitting coefficients.
    See e.g. the following studies
    Wilczak, J. M., S. P. Oncley, and S. A. Stage, 2001: Sonic Anemometer Tilt Correction Algorithms. 
    Boundary-Layer Meteorology, 99, 127–150, https://doi.org/10.1023/A:1018966204465.
    
    Rannik, Ü., and Coauthors, 2020: Impact of coordinate rotation on eddy covariance fluxes at complex sites. 
    Agricultural and Forest Meteorology, 287, 107940, https://doi.org/10.1016/j.agrformet.2020.107940.
    Args:
        u (Series): 30-min averaged unrotated u component of wind vector
        v (Series): 30-min averaged unrotated v component of wind vector
        w (Series): 30-min averaged unrotated w component of wind vector
    Returns:
        (ndarray): fit coefficients b1 and b2
        (float64): fit coefficient b0
        (float64): mean squared error describing the goodness of fit
        (float64): coefficient of determination describing the goodness of fit
    """

    # X = [u,v]
    # X = u + v
    # X = np.concatenate((np.array(u),np.array(v)))
    X = []
    for i in range(len(u)):
        X.append([u[i],v[i]])
    y = w

    # create linear regression object
    reg = linear_model.LinearRegression()
  
    # train the model
    reg.fit(X, y)
    
    # output
    coef = reg.coef_
    inte = reg.intercept_
    MSE = metrics.mean_squared_error(y,reg.predict(X))
    R2 = metrics.r2_score(y,reg.predict(X))

    return coef,inte,MSE,R2





def XCorr(dfin,vrrf,vrs,lglm=None,scaletype='none',hpflag=False,cutoff=1/(5*60),fs=10):
    """Calculates cross-covariance between variables.
    Args:
        dfin (dataframe): dataframe containing data.
        vrrf (str): Reference variable against which the cross-covariance is calculated.
        vrs (list): Variables for which cross-covariance is calculated against vrrf.
        lglm (list, optional): Lag limits over which the cross-covariance is calculated. If empty, then no limits. Defaults to None.
        scaletype (str, optional): Type of scaling applied to cross-covariance. Should be one of the following: 'none', 'biased', 'corr' or 'unbiased'. Defaults to 'none'.
        hpflag (bool, optional): Flag for high-pass filtering the time series before calculating cross-covariances. Defaults to False.
        cutoff ([type], optional): Cut-off frequency for the high-pass filtering. Defaults to 1/(5*60).
        fs (int, optional): Sampling frequency of the data. Defaults to 10.
    Returns:
        dataframe: Cross-covariances
    """    

    df = dfin.copy()

    xcmode = 'same'

    lagged_correlation = pd.DataFrame()   

    if lglm is None:
        lglm = [-len(df),len(df)]
    
    if lglm[0]<-len(df):
        lglm[0]=-len(df)
    if lglm[1]>len(df):
        lglm[1]=len(df)

    # filling NaN with time series median
    # cross-correlation cannot be calculated if time
    # series contain NaN
    df = df.fillna(df.median(numeric_only=True))



    x = df[vrrf]-np.mean(df[vrrf])
    if hpflag:
        # high-pass filtering
        x = butter_highpass_filter(x, cutoff, fs, order=5)

    
    lags = signal.correlation_lags(x.size, x.size, mode=xcmode)
    msk = (lags>=lglm[0])*(lags<=lglm[1])
    lagged_correlation['lag'] = lags[msk]
        
         

    for vr in vrs:
        

        y = df[vr]-np.mean(df[vr])
        if hpflag:
            # high-pass filtering
            y = butter_highpass_filter(y, cutoff, fs, order=5)

        if not y.isnull().all():
            correlation = signal.correlate(y,x, mode=xcmode,method='fft')
            
            if scaletype=='none':
                scale = 1
            elif scaletype=='biased':
                scale = 1/len(x)
            elif scaletype=='corr':
                if (len(x)*np.std(x)*np.std(y))>0:
                    scale = 1/(len(x)*np.std(x)*np.std(y))
                else:
                    scale = np.nan
            elif scaletype=='unbiased':
                scale = (len(x)-np.abs(lags))
                scale[scale<=0] = 1
                scale = 1/scale



            correlation = correlation*scale
            lagged_correlation[vr] = correlation[msk]
        else:
            lagged_correlation[vr] = np.nan*np.ones(np.shape(lagged_correlation['lag']))

        # plt.plot(lags,correlation)
        # plt.show()

    return lagged_correlation


def InstNoise(df,vrs,unitsin=dict()):
    """Estimates instrumental noise (1-sigma white noise) from high frequency time series following Lenschow et al. (2000).
        See also Mauder et al. (2013) and Rannik et al. (2016).
        Lenschow, D. H., V. Wulfmeyer, and C. Senff, 2000: Measuring second- through fourth-order moments in noisy data. 
        Journal of Atmospheric and Oceanic Technology, 17, 1330–1347, https://doi.org/10.1175/1520-0426(2000)017<1330:MSTFOM>2.0.CO;2.
        Mauder, M., M. Cuntz, C. Druee, A. Graf, C. Rebmann, H. P. Schmid, M. Schmidt, and R. Steinbrecher, 2013: A strategy 
        for quality and uncertainty assessment of long-term eddy-covariance measurements. Agricultural and Forest Meteorology, 
        169, 122–135, https://doi.org/10.1016/j.agrformet.2012.09.006.
        Rannik, Ü., O. Peltola, and I. Mammarella, 2016: Random uncertainties of flux measurements by the eddy covariance 
        technique. Atmospheric Measurement Techniques, 9, 5163–5181, https://doi.org/10.5194/amt-9-5163-2016.
    Args:
        df (dataframe): High-frequency data.
        vrs (list): Variables for which the noise is estimated
        unitsin (dict, optional): Units for the data. Defaults to dict().
    Returns:
        dataframe: time series variances decomposed to signal and noise variances. Includes also signal-to-noise ratios.
    """    

    noise = dict()
    units = dict()

    for vr in vrs:
        if vr in df.columns:
            if not np.all(df[vr].isna()):
                xcdf = XCorr(df,vr,[vr],lglm=[0,3],scaletype='biased')
                a, b = np.polyfit(xcdf['lag'][1:],xcdf[vr][1:], 1)
                if b>=0 and b<=xcdf[vr][0]:
                    noise.update({vr+'_noise_var': xcdf[vr][0]-b})
                    noise.update({vr+'_signal_var': b})
                    noise.update({vr+'_SNR': b/(xcdf[vr][0]-b)})
                elif b<0:
                    noise.update({vr+'_noise_var': xcdf[vr][0]})
                    noise.update({vr+'_signal_var': np.nan})
                    noise.update({vr+'_SNR': np.nan})
                else:
                    noise.update({vr+'_noise_var': np.nan})
                    noise.update({vr+'_signal_var': xcdf[vr][0]})
                    noise.update({vr+'_SNR': np.nan})
            else:
                noise.update({vr+'_noise_var': np.nan})
                noise.update({vr+'_signal_var': np.nan})
                noise.update({vr+'_SNR': np.nan})


            if vr in unitsin.keys():
                units.update({vr+'_noise_var': '(' + unitsin[vr] + ')^2'})
                units.update({vr+'_signal_var': '(' + unitsin[vr] + ')^2'})
                units.update({vr+'_SNR': '-'})
        # else:
        #     noise.update({vr+'_noise': np.nan})
        #     noise.update({vr+'_signal': np.nan})
        #     noise.update({vr+'_SNR': np.nan})


        # plt.plot(xcdf['lag'],xcdf[vr])
        # plt.plot(range(0,5,1),a*(range(0,5,1))+b)
        # plt.show()

    noisedf = pd.DataFrame([noise])
    # noisedf = noisedf.append(noise, ignore_index=True)
    return noisedf,units



def EstimateLag(df,vrrf,vrs,lglm,hpflag=False,cutoff=1/(5*60),fs=10):
    """Estimates time lag between variables from cross-covariance maximum.
    Args:
        df (dataframe): High-frequency EC raw data.
        vrrf (str): Reference variable against which time lags are estimated.
        vrs (list): Variables for which time lags are estimated.
        lglm (list): Limits for time lag values.
        hpflag (bool, optional): Flag for high-pass filtering time series before time lag estimation. Defaults to False.
        cutoff ([type], optional): Cut-off frequency for the high-pass filtering. Defaults to 1/(5*60).
        fs (int, optional): Data sampling frequency. Defaults to 10.
    Returns:
        dataframe: Time lags and cross-correlation maxima.
    """    

    lags = dict()
    units = dict()
    for vr in vrs:
        if vr in df.columns:
            xcdf = XCorr(df,vrrf,[vr],lglm,'corr',hpflag=hpflag,cutoff=cutoff,fs=fs)
            mxval = np.max(np.abs(xcdf[vr]))
            lag = xcdf['lag'][np.abs(xcdf[vr])==mxval].reset_index(drop = True)
            
            if not lag.empty:
                lags.update({vr+'_lag': lag[0]})
                lags.update({vr+'_corr': mxval})
            else:
                lags.update({vr+'_lag': np.nan})
                lags.update({vr+'_corr': np.nan})

            
            
            units.update({vr+'_lag': 'points'})
            units.update({vr+'_corr': '-'})
        # else:
        #     lags.update({vr+'_lag': np.nan})
        #     lags.update({vr+'_corr': np.nan})

        # plt.plot(xcdf['lag'],xcdf[vr])
        # plt.show()
    
    lagsdf = pd.DataFrame([lags])
    return lagsdf,units


def butter_highpass(cutoff, fs, order=5):
    """Creates Butterworth filter.
    Args:
        cutoff (int): Cut-off frequency of the filter.
        fs (int): Sampling frequency of the data.
        order (int, optional): Order of the filter. Defaults to 5.
    Returns:
        ndarray, ndarray: Numerator (b) and denominator (a) polynomials of the IIR filter
    """    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """Applies the Butterworth filter to data.
    Args:
        data (Series): Time series that is filtered.
        cutoff (int): Cut-off frequency of the filter.
        fs (int): Sampling frequency of the data.
        order (int, optional): Order of the filter. Defaults to 5.
    Returns:
        Series: High-pass filtered time series.
    """    
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y



def RetrieveFMIObs(latin,lonin,ts,te,deltat,distlm=50000,interpolateflag=False,varsin=None,dattype='weather'):
    # download FMI observations in a daily loop
    # long time series cannot be downloaded at once
    
    time = pd.date_range(start=ts,end=te,freq=deltat).to_series()
    time.reset_index(drop=True,inplace=True)


    # time.sort_values(inplace=True)
    # time.reset_index(inplace=True)

    dates = time.dt.date.unique()

    df = pd.DataFrame()
    for date in dates:
        print(date)
        
        time2 = time[date==time.dt.date]
        

        ts = np.min(time2).to_pydatetime()
        te = np.max(time2).to_pydatetime()

        datout,unitsout,site = RetrieveFMIObservations(latin,lonin,ts,te,distlm=distlm,varsin=varsin,dattype=dattype)

        df = df.append(datout, ignore_index = True)

        
    if 'time' in df.columns:
        # df.set_index('time', drop=False, append=False, inplace=True, verify_integrity=False)
        # df = df.sort_index()
        df.sort_values('time',inplace=True)

        # matching data frequency with input time
        dfout = pd.DataFrame()
        dfout['time'] = time
        if interpolateflag:
            vars = list(df.columns)
            vars.remove('time')

            # interp1 does not work with datetime => converting to float64
            tin = np.array(df['time']).astype('float64')
            tout = np.array(time).astype('float64')
            for vr in vars:
                if vr=='P_RAIN_FMI':
                    # not 100 % sure if correct
                    linfit = interp1d(tin, df[vr],bounds_error=False,kind='previous')
                    val = linfit(tout)
                    
                else:
                    # linear interpolation for other variables
                    linfit = interp1d(tin, df[vr],bounds_error=False)
                    val = linfit(tout)
                dfout[vr] = val
        else:
            dfout = dfout.merge(df,how='outer',on='time')
    else:
        dfout = pd.DataFrame()
        dfout['time'] = time
        unitsout = dict()
        unitsout.update({'time':''})
        site = None

    dfout.sort_values('time',inplace=True)
    dfout.reset_index(inplace=True)


    return dfout,unitsout,site
    



def RetrieveFMIObservations(latin,lonin,ts,te,distlm=50000,varsin=None,site=None,dattype='weather'):

    wgs84_geod = Geod(ellps='WGS84') #Distance will be measured on this ellipsoid - more accurate than a spherical method

    # latin = cconfig['site_info']['latitude']
    # lonin = cconfig['site_info']['longitude']
    # time = df['time']
    
    # end_time = np.max(time).to_pydatetime()-dt.timedelta(days=30)
    # start_time = np.min(time).to_pydatetime()-dt.timedelta(days=30)
    # # start_time = np.max(time).to_pydatetime()-dt.timedelta(days=31)
    # end_time = dt.datetime.utcnow()-dt.timedelta(days=30)
    # start_time = end_time-dt.timedelta(days=1)

    # Convert times to properly formatted strings
    start_time = ts.isoformat(timespec="seconds") + "Z"
    # -> 2020-07-07T12:00:00Z
    end_time = te.isoformat(timespec="seconds") + "Z"
    # -> 2020-07-07T13:00:00Z
    
    # stored query
    if dattype == 'weather':
        obs = download_stored_query("fmi::observations::weather::multipointcoverage",
                                    args=['bbox=' + f"{lonin-1:.0f}" + ',' + f"{latin-1:.0f}" + ',' + f"{lonin+1:.0f}" + ',' + f"{latin+1:.0f}",
                                        "starttime=" + start_time,
                                        "endtime=" + end_time,"timeseries=True"])
    elif dattype == 'radiation':
        radsites = ['102035','101932','101920','107201','101787','101756','101586','101339','101104','100968','101030','101004','100908','100929']
        sitestr = ''
        for rsite in radsites:
            sitestr = sitestr + '&fmisid=' + rsite
        obs = download_stored_query("fmi::observations::radiation::multipointcoverage",
                            args=[sitestr,
                                  "starttime=" + start_time,
                                  "endtime=" + end_time,"timeseries=True"])
    # obs = download_stored_query("fmi::observations::weather::multipointcoverage",
    #                             args=["bbox=18,55,35,75",
    #                                 "starttime=" + start_time,
    #                                 "endtime=" + end_time,"timeseries=True"])

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

    return datout,unitsout,site
