# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:45:02 2023

@author: 03081268
"""

#testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ffile = r'c:\data\GFB\FI-Var\proc\FI-Var_2013_2022.dat'

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
    
    if not flags:
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
        print(window)
        fdat = DiurnalPattern(dout, method=method, deltadays=window, Nmin=Nmin, deltat=deltat)
        ix = np.where(dout.isna())
        #ix = np.where((dout.isna()==True) & (fdat.isna()==False))[0]
        
        dout.iloc[ix] = fdat.iloc[ix]
        flags.iloc[ix] = window 
        
        window += 2
        
        # bool
        gaps = dout.isna().any()
        print(len(np.where(dout.isna())[0]))
    
    return dout, flags

#%% get data
flx = pd.read_csv(ffile, sep=';', decimal='.')
t = pd.to_datetime(flx[['Year', 'Month', 'Day', 'Hour', 'Minute']])
flx.index = t

#%%
dat = flx.loc[flx.index < '2015-01-01'] 

yin = dat['Tair']

out, flags = InterpolateShortGaps(yin, N=10)

out2, flags = MDC(out, method='median', deltat='30min', flags=flags)

fig, ax = plt.subplots(2,1, sharex=True)
#ax[0].plot(dint, '.-', label='dint')
ax[0].plot(out2, '.-', label='MDC')
ax[0].plot(out, '.-', label='lin')
ax[0].plot(yin, '.-', label='obs')
ax[0].legend()

ax[1].plot(flags, '.-')

#%%
met = flx.copy()
met.rename(columns={'Precip':'Prec', 'diffPAR':'diffPar', 'Tsoil1':'Tsoil', 'wsoil': 'Wsoil'}, inplace=True)
met = met[['Rn', 'Rg', 'Par', 'diffPar', 'U', 'u_star', 'Tair', 'RH', 'CO2', 'LWin', 'Prec',
           'Tsoil', 'Wsoil']]

met['P'] = 101300.0
met = met[~met.index.duplicated(keep='first')]
#%%
from src.utils import create_forcingfile

lat = 67.747
lon = 29.618
forc, flags = create_forcingfile(met, 'output_file',lat, lon, met_data=None, timezone=+2.0,
                                 CO2_constant=False, short_gap_len=5)

#%% K-NN
from sklearn.metrics import mean_squared_error 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


d_f = met['diffPar'] / (met['Par'] + 1e-8)
d_f[d_f<0.2] = 0.2

d_f[forc['Zen'] <= 0] = 1.0
d_f[d_f>1.0] = 1.0

ix = np.where((d_f.isna() == False) & (met['Par']>10))[0]

Y = d_f
X = forc[['doy', 'Zen', 'Tair', 'H2O', 'U', 'Rg', 'Prec']]
X['doy'] = X['doy'] + X.index.hour.values / 24 + X.index.hour.values / (24 * 60)

X = X[['doy', 'Zen', 'Rg']]#, 'H2O']]

rf = RandomForestRegressor(n_estimators=10)
rf_fit = rf.fit(X.iloc[ix].values, Y.iloc[ix].values)

Y_pred = rf.predict(X.values)
Y_pred[forc['Zen']<=0] = 1.0

plt.figure()
plt.plot(Y.iloc[ix], 'r.-', label='obs')
#plt.plot(forc['f_d'].iloc[ix], 'g.-', label='mod')
#plt.plot(X.index, pred_y, '-', label='KNN')
plt.plot(X.index[ix], Y_pred[ix], '-', label='RF')
plt.ylabel('diffuse fraction')
plt.legend()

plt.figure()
plt.plot(Y.iloc[ix], Y_pred[ix], 'ro', alpha=0.1)
plt.ylabel('RF')
plt.xlabel('obs')

#ixp = np.setdiff1d(np.arange(0, len(Y)), ix)
ixp = np.where((d_f.isna()) & (met['Par']>10))[0]

fig, ax = plt.subplots(1,2)

ax[0].hist(Y.iloc[ix], bins=20, alpha = 0.5, label='obs')
ax[0].hist(Y_pred[ix], bins=20, alpha = 0.5, label='RF - train')
ax[0].set_ylabel('counts')
ax[0].set_xlabel('diffuse fraction')
ax[0].legend()

ax[1].hist(Y.iloc[ix], bins=20, alpha = 0.5, label='obs - train')
ax[1].hist(Y_pred[ixp], bins=20, alpha = 0.5, label='RF - predict')
ax[1].set_ylabel('counts')
ax[1].set_xlabel('diffuse fraction')
ax[1].legend()

#%%
from src.utils import gap_fill_rf

d_f = met['diffPar'] / (met['Par'] + 1e-8)
d_f[d_f<0.2] = 0.2

d_f[forc['Zen'] <= 0] = 1.0
d_f[met['Par'] < 10] = 1.0
d_f[d_f>1.0] = 1.0

#ix = np.where((d_f.isna() == False) & (met['Par']>10))[0]
ix = np.where((d_f.isna() == False))[0]

Y = d_f
X = forc[['doy', 'Zen', 'Tair', 'H2O', 'U', 'Rg', 'Prec']]
X['doy'] = X['doy'] + X.index.hour.values / 24 + X.index.hour.values / (24 * 60)

X = X[['doy', 'Zen', 'Rg']].values#, 'H2O']]
X = X.values
Y = Y.values

Ygf, Ypred = gap_fill_rf(Y, X, ix)

#%% estimate emi_sky and LWin
#: [W m\ :sup:`-2` K\ :sup:`-4`\ ], Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.6697e-8

LWin = met['LWin']
Tk = met['Tair'] + 273.15

emi_sky = LWin / (STEFAN_BOLTZMANN * Tk**4)
emi_sky[emi_sky > 1.0] = 1.0
ix = np.where(emi_sky.isna()==False)

plt.figure()
plt.plot(emi_sky, '.-')

Y = emi_sky.values
X = forc[['doy', 'Zen', 'Tair', 'H2O', 'U', 'Rg', 'Prec']]
X['doy'] = X['doy'] + X.index.hour.values / 24 + X.index.hour.values / (24 * 60)

X = X[['doy', 'Tair', 'H2O', 'Rg']].values

Ypred = gap_fill_rf(Y, X, ix, figtitle='emi_sky')

LWin_pred = Ypred * STEFAN_BOLTZMANN * Tk**4

plt.figure()
plt.plot(forc['LWin'], '.-', alpha=0.5, label='mod')
plt.plot(LWin_pred, '.-', alpha=0.9, label='pred')
plt.plot(LWin, '.-', alpha=0.9, label='obs')
plt.legend()

