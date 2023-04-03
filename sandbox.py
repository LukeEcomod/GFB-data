# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:45:02 2023

@author: 03081268
"""

#testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn import preprocessing

from src.utils import rf_crossvalidate


def clear_sky_radiation(doy, zen):
    # clear sky Global radiation at surface
    So = 1367.0
    Rg0 = np.maximum(0.0, (So * (1.0 + 0.033 * np.cos(2.0 * np.pi * (np.minimum(doy, 365) - 10) / 365)) * np.cos(zen)))
    
    return Rg0

ffile = r'c:\data\GFB\FI-Var\proc\FI-Var_forcing_2013-2022.dat'

dat = pd.read_csv(ffile, sep=';')
t = pd.to_datetime(dat[['Year', 'Month', 'Day', 'Hour', 'Minute']])
dat.index = t

cols = list(dat.columns)

x = dat[['doy', 'Zen', 'Prec', 'P', 'Tair', 'H2O', 'dirPar', 'diffPar']]
x['Par'] = x['dirPar'] + x['diffPar']

#doy at UTC +2
decimal_doy = x['doy'].values + x.index.hour.values / 24 + x.index.minute.values / (24*60)
decimal_doy -= 2.0/24

x['Rg0'] = clear_sky_radiation(decimal_doy, x['Zen'].values)

# create daily data

dave = x.resample('D').mean()
dave['Prec'] = x['Prec'].resample('D').sum()

dmin = x[['Tair', 'H2O']].resample('D').min()
dmin.columns = [c + '_min' for c in dmin.columns]

dmax = x[['Tair', 'H2O']].resample('D').max()
dmax.columns = [c + '_max' for c in dmax.columns]

ddata = pd.concat([dave, dmin, dmax], axis=1)
ddata = ddata.resample('30min').ffill()
ddata = ddata.reindex(index=x.index)
ddata = ddata.fillna(method='ffill')

ddata['Zen'] = x['Zen']
ddata['Rg0'] = x['Rg0']
#ddata['Par'] = ddata['dirPar'] + ddata['diffPar']

#%% test

v = ['Tair', 'H2O']
#v = 'Tair'

ixt = np.where(x.index.year<2016)[0]
ixp = np.where(x.index.year>2016)[0]

#pcols = ['doy', 'Zen', 'Prec', 'P', 'Tair', 'Tair_min', 'Tair_max', 'H2O', 'H2O_min', 'H2O_max']#, 'Par']
pcols = ['Zen', 'Tair', 'Tair_min', 'Tair_max', 'H2O', 'H2O_min', 'H2O_max'] #, 'Par']

Y = x[v].iloc[ixt].values
#Y = Y.reshape(-1, 1)
X = ddata.iloc[ixt].values
t = x.index[ixt]

rf, scaler_X, scaler_Y, cv_score, score = rf_crossvalidate(X, Y, test_size=0.3, nfolds=5, n_estimators=10)


Yp = x[v].iloc[ixp].values
#Yp = Yp.reshape(-1, 1)
Xp = ddata.iloc[ixp].values
tp = x.index[ixp]

# prediction to train set
Ytpred = scaler_Y.inverse_transform(rf.predict(scaler_X.transform(X)))
print('score (R2) in train set: %.2f' % rf.score(scaler_X.transform(X), 
                                               scaler_Y.transform(Y))
      )

# prediction to independent test set
Ypred = scaler_Y.inverse_transform(rf.predict(scaler_X.transform(Xp)))
print('score (R2) in test set: %.2f' % rf.score(scaler_X.transform(Xp),
                                              scaler_Y.transform(Yp))
      )

#v = ['Tair']
for k in range(len(v)):
    
    plt.figure()
    plt.plot(t, Y[:,k], '-', label='obs-train')
    plt.plot(t, Ytpred[:,k], '-', label='pred-train')
    plt.plot(tp, Yp[:,k], '-', label='obs-test')
    plt.plot(tp, Ypred[:,k], '-', label='pred-test')
    plt.title(v[k])
    plt.legend()
    
    fig, ax = plt.subplots(1,2)
    ax[0].set_title(v[k] + ' train set')
    
    ax[0].plot(Y[:,k], Ytpred[:,k], '.', alpha=0.3, label='train')
    ax[0].set_xlabel('obs')
    ax[0].set_ylabel('pred')
    
    p = np.polyfit(Y[:,k], Ytpred[:,k], 1)
    xx = np.array([min(Y[:,k]), max(Y[:,k])])
    fit = p[1] + p[0] * xx
    ax[0].plot(xx, fit, '-', label='%.2f' %p[0])
    ax[0].legend()
    
    ax[1].set_title(v[k] + ' test set')
    
    ax[1].plot(Yp[:,k], Ypred[:,k], '.', alpha=0.3, label='train')
    ax[1].set_xlabel('obs')
    ax[1].set_ylabel('pred')
    
    p = np.polyfit(Yp[:,k], Ypred[:,k], 1)
    xx = np.array([min(Y[:,k]), max(Y[:,k])])
    fit = p[1] + p[0] * xx
    ax[1].plot(xx, fit, '-', label='%.2f' %p[0])
    ax[1].legend()
    
    
    # plt.figure()
    # plt.plot(t, Y[:,k], '-', label='obs-train')
    # plt.plot(t, Ytpred[:,k], '-', label='pred-train')
    # plt.plot(tp, Yp, '-', label='obs-test')
    # plt.plot(tp, Ypred, '-', label='pred-test')
    # plt.title(v)
    # plt.legend()
    
    # fig, ax = plt.subplots(1,2)
    # ax[0].set_title(v + ' train set')
    
    # ax[0].plot(Y, Ytpred, '.', alpha=0.3, label='train')
    # ax[0].set_xlabel('obs')
    # ax[0].set_ylabel('pred')
    
    # p = np.polyfit(Y, Ytpred, 1)
    # xx = np.array([min(Y), max(Y)])
    # fit = p[1] + p[0] * xx
    # ax[0].plot(xx, fit, '-', label='%.2f' %p[0])
    # ax[0].legend()
    
    # ax[1].set_title(v + ' test set')
    
    # ax[1].plot(Yp, Ypred, '.', alpha=0.3, label='train')
    # ax[1].set_xlabel('obs')
    # ax[1].set_ylabel('pred')
    
    # p = np.polyfit(Yp, Ypred, 1)
    # xx = np.array([min(Y), max(Y)])
    # fit = p[1] + p[0] * xx
    # ax[1].plot(xx, fit, '-', label='%.2f' %p[0])
    # ax[1].legend()