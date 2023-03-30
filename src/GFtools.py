# -*- coding: utf-8 -*-
"""
Copied from pyECPC 27.3.2023
https://github.com/Luke-GHG-Data/PyECPC/blob/Olli_dev/PyECPC_utilities/GFtools.py

Author: Olli Peltola
-----
Created Date: Wednesday February 1st 2023
Last Modified:
-----
"""
    

import datetime as dt
import PyECPC_utilities.tools as tools
import pandas as pd
import numpy as np
import logging
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict,train_test_split,KFold
from sklearn.feature_selection import RFECV
import sklearn.metrics
from scipy.stats import pearsonr,ttest_1samp
from sklearn.inspection import permutation_importance
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from eli5.sklearn import PermutationImportance as PermImp



logger = logging.getLogger("ecpc_log")


def Preprocess(df,drvvr,vr,fsmethod=None,fstol=None,n_train=10,eval_frac=0.15,seed=1000):
    """Splits data to train and test sets and selects predictors.
    Args:
        df (DataFrame): Time series of predictors and target variable.
        drvvr (list): List of predictor names.
        vr (str): Target variable.
        fsmethod (str, optional): Method used to select the predictors. Defaults to None.
        fstol (float, optional): Tolerance for a statistical metric used when selecting predictors. Defaults to None.
        n_train (int, optional): Amount of training sets. Defaults to 10.
        eval_frac (float, optional): Fraction of data used for creating test set. Defaults to 0.15.
        seed (int, optional): Seed for random number generator. Defaults to 1000.
    Returns:
        list: List of predictor names.
        dict: Predictor importance scores.
        dict: indeces for train and test datasets.
    """

    # dividing data into training and testing data sets based on real gap distributions
    # gap distribution
    gap_pmf = GapDistribution(df[vr])

    # test data
    test_vr = sample_artificial_gaps(df[vr],gap_pmf,eval_frac=eval_frac,seed=seed)
    test_indices = test_vr.isna() & ~df[vr].isna()

    # Remove test set
    trnval_indices = ~test_vr.isna() & ~df[vr].isna()
    
    # n_train models that are used for estimating random uncertainty
    train_valid_pairs = []
    if n_train>0:
        trns = []
        for i in range(n_train):trns.append(pd.Series(index=trnval_indices.index,data=False))
        istrt = 0
        wind = 11
        while istrt<len(trnval_indices):
            for i in range(n_train):
                if istrt<len(trnval_indices):
                    iend = np.min([istrt+wind,len(trnval_indices)])
                    trns[i][istrt:iend] = trnval_indices[istrt:iend]
                    istrt = iend

        for i in range(n_train):
            val = trnval_indices & ~trns[i]
            train_valid_pairs.append((trns[i], val))

    # all data (except test data) are used for the last model that is used for the prediction
    train_valid_pairs.append((trnval_indices,pd.Series(index=trnval_indices.index,data=False)))

    
    # id = shuffle(trnval_indices)
    # idi = id.index
    # id.reset_index(drop=True,inplace=True)
    # lind = (id.cumsum()/np.sum(~df[vr].isna())).gt(eval_frac).idxmax()
    # val = pd.Series(index=trnval_indices.index,data=False)

    # tmp = pd.Series(index=idi,data=id.to_numpy())
    # val[idi[0:lind]] = tmp[0:lind]
    # trn = ~df[vr].isna() & ~val & ~test_vr.isna()
    # train_valid_pairs.append((trn, val))

    tsts = list()
    for trn,val in train_valid_pairs:
        # the sum should match with the amount of observations
        tsts.append(np.sum(test_indices)+np.sum(trn)+np.sum(val))   
        
        
    split = dict()
    split['test'] = test_indices
    split['trainval'] = train_valid_pairs

        

    # # feature selection based on train/valid data set
    if fsmethod is not None:
        drvvr,imp = FeatureSelection(df[~test_indices], drvvr, vr,method=fsmethod,tol=fstol)
    else:
        imp = dict()
        imp['mean'] = [np.nan]*len(drvvr)
        imp['std'] = [np.nan]*len(drvvr)


    return drvvr,imp,split

def EstimateSystematicError(mods,splits,obs,sig_level=None,deltadays=None,min_periods=None):
    """Generates estimates for systematic errors (biases) by using gapfilling models and test datasets.
    Args:
        mods (dict): Gapfilling models.
        splits (dict): Indeces for train and test data sets.
        obs (Series): Time series of target variable observations.
        sig_level (float, optional): Significance level. Defaults to None.
        deltadays (int, optional): Window width within which biases are estimated. Defaults to None.
        min_periods (int, optional): Minimum amount of data needed for estimating bias. Defaults to None.
    Returns:
        DataFrame: DataFrame containing estimates for biases for each model type in mods
    """    

    name = obs.name
    ntest = len(mods)
    modeltypes = list(mods[0].keys())
    popmean = 0

    if sig_level is None:
        sig_level=0.05
    if deltadays is None:
        deltadays=3
    if min_periods is None:
        min_periods = 10

    def run_test(x, popmean):
        x = x[~x.isna()]
        tscore, pvalue = ttest_1samp(x, popmean)
        return pvalue


    dfse = pd.DataFrame()
    for type in modeltypes:


        # biasdf contains the model residual for test sets
        biasdf = pd.DataFrame()
        for j in range(ntest):
            
            # extracting the model prediction
            if (type!='MDS') & (type!='MDSss') & (type!='ensemble'):
                pred = mods[j][type][len(mods[j][type])-2]['pred']
            else:
                pred = mods[j][type]['pred']

            apu = pred[splits[j]['test']]-obs
            apu = apu[~apu.isna()]
            apu.name = name
            apu = apu.reset_index()
            apu = apu.rename(columns={'index':'time'})
            biasdf = pd.concat([biasdf,apu])

        # averaging overlapping test sets
        biasdf = biasdf.groupby('time').mean()

        biasdf = biasdf.reindex(index=obs.index)
        # average of model residual over deltadays
        biasdf2 = biasdf.rolling(dt.timedelta(days=deltadays),center=True,min_periods=min_periods).mean()

        # t-test for testing whether the average of model residual is significantly different from zero
        if sig_level<1:
            ttest = biasdf.rolling(dt.timedelta(days=deltadays),center=True,min_periods=min_periods).agg(run_test, popmean = popmean)
        else:
            ttest = pd.Series(index=obs.index,data=0)

        out = pd.DataFrame(index=obs.index)
        out[name] = 0
        # periods when model residual is significantly different from zero => systematic error
        out[ttest<sig_level] = biasdf2[ttest<sig_level].abs()
        dfse[type] = out

    return dfse

def Gapfill(df,drvvr,vr,vrranderr,models,vrsyserr=None,tol=None,subsample=None,MDSdrvvr=None,dfse=None,hvdays=None):
    """Fills gaps in target variable time series using gapfilling models.
    Args:
        df (DataFrame): Time series of target variable.
        drvvr (list): List of predictor names.
        vr (str): Target variables.
        vrranderr (str): Target variable random error.
        models (dict): Gapfilling models.
        vrsyserr (str, optional): Target variable random error. Defaults to None.
        tol (dict, optional): Predictor tolerances used by the MDS algorithm. Defaults to None.
        subsample (dict, optional): Boolean for each predictor for subsampling (MDS algorithm). Defaults to None.
        MDSdrvvr (list, optional): List of MDS algorithm predictors. Defaults to None.
        dfse (DataFrame, optional): Time series of gapfilling model systematic errors. Defaults to None.
        hvdays (list, optional): List of datetimes for harvest at the site. Defaults to None.
    Returns:
        DataFrame: Gapfilled time series and uncertainties.
    """    
    
    dfout = pd.DataFrame(index=df.index)
    types = list(models.keys())
    # making sure that ensemble model is the last in list
    if 'ensemble' in types:
        types.remove('ensemble')
        types.append('ensemble')

    yobs = df[vr]
    yobsre = df[vrranderr]
    if vrsyserr is None:
        yobsse = pd.Series(index=df.index,data=0)
    else:
        yobsse = df[vrsyserr]
    if dfse is None:
        dfse = pd.DataFrame(index=df.index)
        for type in types:
            dfse[type] = 0

    # taking into account only those harvest days that are within the analysed period     
    hvdays = [hvd for hvd in hvdays if ((hvd>df.index[0]) & (hvd<df.index[-1]))]
    if hvdays is None:
        tper = [[df.index[0],df.index[-1]]]
    else:
        hvdays.append(df.index[0].to_pydatetime())
        hvdays.append(df.index[-1].to_pydatetime())
        hvdays.sort()

        tper = list()
        for i in range(len(hvdays)-1):
            if (hvdays[i]>=df.index[0]) & (hvdays[i]<=df.index[-1]):
                tper.append([pd.Timestamp(hvdays[i]),pd.Timestamp(hvdays[i+1])])

    # Boolean for gaps in time series
    dfout['gap'] = False
    dfout.loc[yobs.isna(),'gap'] = True

    # predictions
    tmp = pd.DataFrame(index=df.index)
    tmp4 = pd.DataFrame(index=df.index)
    tmp5 = dict()
    for type in types:
        if type!='ensemble':
            if (type!='MDS') & (type!='MDSss'):
                tmp2 = pd.DataFrame(index=df.index)
                keys = list(models[type].keys())
                for i in range(len(keys)):
                    key = keys[i]
                    mod = models[type][key]
                    if key!='ensemble':
                        ypred = MLpredict(mod['model'],df,drvvr)
                        tmp2[i] = ypred
                dfout[vr+'_gf_'+type+'_ensemble'] = yobs
                dfout[vr+'_gf_'+type] = yobs
                dfout.loc[yobs.isna(),vr+'_gf_'+type] = tmp2.loc[yobs.isna(),len(tmp2.columns)-1]

                # random error
                dfout[vr+'_gf_'+type+'_rand_err'] = yobsre
                dfout.loc[yobs.isna(),vr+'_gf_'+type+'_rand_err'] = tmp2.loc[yobs.isna(),0:len(tmp2.columns)-2].std(axis=1)

                # systematic error
                dfout[vr+'_gf_'+type+'_sys_err'] = yobsse
                dfout.loc[yobs.isna(),vr+'_gf_'+type+'_sys_err'] = dfse.loc[yobs.isna(),type]

                tmp[type] = dfout[vr+'_gf_'+type]
                tmp4[type] = dfout[vr+'_gf_'+type+'_sys_err']
                tmp5[type] = tmp2.loc[:,0:len(tmp2.columns)-2]
            elif type=='MDS':
                if (tol is not None) & (MDSdrvvr is not None):
                    dfMDS = pd.DataFrame(index=df.index,columns=[vr+'_gf',vr+'_gf_rand_err'])
                    for ts,te in tper:
                        dfMDStmp = pd.DataFrame(index=df.index)

                        for drv in MDSdrvvr:
                            dfMDStmp[drv] = df[drv]
                        dfMDStmp[vr] = df[vr]
                        dfMDStmp['time'] = pd.to_datetime(df.index)
                        dfMDStmp[vr+'_gf'] = df[vr]
                        dfMDStmp[vr+'_gf_rand_err'] = np.nan
                        dfMDStmp = dfMDStmp[ts:te]
                        
                        dfMDStmp = MDS(dfMDStmp,vr,vr+'_gf',MDSdrvvr,tol,method='mean',vrunc=vr+'_gf_rand_err')
                        dfMDS.loc[dfMDStmp.index,vr+'_gf'] = dfMDStmp[vr+'_gf']
                        dfMDS.loc[dfMDStmp.index,vr+'_gf_rand_err'] = dfMDStmp[vr+'_gf_rand_err']


                    dfout[vr+'_gf_'+type] = yobs
                    dfout.loc[yobs.isna(),vr+'_gf_'+type] = dfMDS.loc[yobs.isna(),vr+'_gf']
                    dfout[vr+'_gf_'+type+'_rand_err'] = yobsre
                    dfout.loc[yobs.isna(),vr+'_gf_'+type+'_rand_err'] = dfMDS.loc[yobs.isna(),vr+'_gf_rand_err']
                    dfout[vr+'_gf_'+type+'_sys_err'] = yobsse
                    dfout.loc[yobs.isna(),vr+'_gf_'+type+'_sys_err'] = dfse.loc[yobs.isna(),type]
                else:
                    dfout[vr+'_gf_'+type] = yobs
                    dfout[vr+'_gf_'+type+'_rand_err'] = yobsre
                    dfout[vr+'_gf_'+type+'_sys_err'] = yobsse
            elif type=='MDSss':
                if (tol is not None) & (MDSdrvvr is not None) & (subsample is not None):
                    dfMDS = pd.DataFrame(index=df.index)
                    for ts,te in tper:
                        dfMDStmp = pd.DataFrame(index=df.index)

                        for drv in MDSdrvvr:
                            dfMDStmp[drv] = df[drv]
                        dfMDStmp[vr] = df[vr]
                        dfMDStmp['time'] = pd.to_datetime(df.index)
                        dfMDStmp[vr+'_gf'] = df[vr]
                        dfMDStmp[vr+'_gf_rand_err'] = np.nan
                        dfMDStmp = dfMDStmp[ts:te]
                        
                        dfMDStmp = MDS(dfMDStmp,vr,vr+'_gf',MDSdrvvr,tol,method='mean',vrunc=vr+'_gf_rand_err',subsample=subsample)
                        dfMDS.loc[dfMDStmp.index,vr+'_gf'] = dfMDStmp[vr+'_gf']
                        dfMDS.loc[dfMDStmp.index,vr+'_gf_rand_err'] = dfMDStmp[vr+'_gf_rand_err']



                    dfout[vr+'_gf_'+type] = yobs
                    dfout.loc[yobs.isna(),vr+'_gf_'+type] = dfMDS.loc[yobs.isna(),vr+'_gf']
                    dfout[vr+'_gf_'+type+'_rand_err'] = yobsre
                    dfout.loc[yobs.isna(),vr+'_gf_'+type+'_rand_err'] = dfMDS.loc[yobs.isna(),vr+'_gf_rand_err']
                    dfout[vr+'_gf_'+type+'_sys_err'] = yobsse
                    dfout.loc[yobs.isna(),vr+'_gf_'+type+'_sys_err'] = dfse.loc[yobs.isna(),type]
                else:
                    dfout[vr+'_gf_'+type] = yobs
                    dfout[vr+'_gf_'+type+'_rand_err'] = yobsre
                    dfout[vr+'_gf_'+type+'_sys_err'] = yobsse
        else:
            # uncertainties of the ensemble model
            keys = list(tmp5.keys())
            mns = pd.DataFrame(index=df.index)
            for i in range(np.shape(tmp5[keys[0]])[1]):
                datML = pd.DataFrame(index=df.index)
                for key in keys:
                    datML[key] = tmp5[key][i]
                mns[i] = datML.mean(axis=1)
            dfout[vr+'_gf_ensemble_rand_err'] = yobsre
            dfout.loc[yobs.isna(),vr+'_gf_ensemble_rand_err'] = mns[yobs.isna()].std(axis=1)
            dfout[vr+'_gf_ensemble_sys_err'] = yobsse
            dfout.loc[yobs.isna(),vr+'_gf_ensemble_sys_err'] = dfse.loc[yobs.isna(),type]
            
    
    dfout[vr+'_gf_ensemble'] = yobs
    dfout.loc[yobs.isna(),vr+'_gf_ensemble'] = tmp[yobs.isna()].mean(axis=1)

    return dfout    

def TestModels(df,drvvr,vr,models,split,metrics=None,tol=None,subsample=None,MDSdrvvr=None,hvdays=None):

    if metrics is None:
        metrics = ['r2','bias','rmse','pr2']
    if hvdays is None:
        tper = [[df.index[0],df.index[-1]]]
    else:
        hvdays.append(df.index[0].to_pydatetime())
        hvdays.append(df.index[-1].to_pydatetime())
        hvdays.sort()

        tper = list()
        for i in range(len(hvdays)-1):
            tper.append([pd.Timestamp(hvdays[i]),pd.Timestamp(hvdays[i+1])])


    yobs = df[vr]

    types = list(models.keys())

    # predictions
    ypreds = dict()
    tmp = pd.DataFrame(index=df.index)
    for type in types:
        if type!='ensemble':
            if (type!='MDS') & (type!='MDSss'):
                ypreds[type] = dict()
                tmp2 = pd.DataFrame(index=df.index)
                keys = list(models[type].keys())
                for i in range(len(keys)):
                    key = keys[i]
                    mod = models[type][key]
                    if key!='ensemble':
                        ypred = MLpredict(mod['model'],df,drvvr)
                        ypreds[type][key] = ypred
                        tmp2[i] = ypred
                ypreds[type]['ensemble'] = tmp2.mean(axis=1)
                tmp[type] = ypreds[type][len(keys)-2]
            elif (type=='MDS'):
                ypreds[type] = pd.Series(index=df.index)
                if (tol is not None) & (MDSdrvvr is not None):

                    for ts,te in tper:
                        dfMDS = pd.DataFrame(index=df.index)
                        for drv in MDSdrvvr:
                            dfMDS[drv] = df[drv]
                        dfMDS[vr] = df[vr]
                        dfMDS['time'] = pd.to_datetime(df.index)
                        # removing test data from MDS estimation
                        dfMDS.loc[split['test'],vr] = np.nan
                        dfMDS[vr+'_gf'] = np.nan
                        dfMDS = dfMDS[ts:te]
                        
                        dfMDS = MDS(dfMDS,vr,vr+'_gf',MDSdrvvr,tol,method='mean',timeisna=dfMDS.loc[split['test'],'time'])
                        ypreds[type][dfMDS.index] = dfMDS[vr+'_gf']

            elif (type=='MDSss'):
                ypreds[type] = pd.Series(index=df.index)
                if (tol is not None) & (MDSdrvvr is not None) & (subsample is not None):
                    for ts,te in tper:
                        dfMDS = pd.DataFrame(index=df.index)
                        for drv in MDSdrvvr:
                            dfMDS[drv] = df[drv]
                        dfMDS[vr] = df[vr]
                        dfMDS['time'] = pd.to_datetime(df.index)
                        # removing test data from MDS estimation
                        dfMDS.loc[split['test'],vr] = np.nan
                        dfMDS[vr+'_gf'] = np.nan
                        dfMDS = dfMDS[ts:te]
                        
                        dfMDS = MDS(dfMDS,vr,vr+'_gf',MDSdrvvr,tol,method='mean',timeisna=dfMDS.loc[split['test'],'time'],subsample=subsample)
                        ypreds[type][dfMDS.index] = dfMDS[vr+'_gf']


                
    ypreds['ensemble'] = tmp.mean(axis=1)

    # calculation of performance metrics against test data
    for type in types:
        keys = list(models[type].keys())
        if 'ensemble' in keys:
            for i in range(len(keys)):
                key = keys[i]
                ypred = ypreds[type][key]
                tstmetrics = CalcMetrics(yobs[split['test']],ypred[split['test']],metrics) 
                models[type][key]['testmetrics'] = tstmetrics
                models[type][key]['pred'] = ypred
        else:
            ypred = ypreds[type]
            tstmetrics = CalcMetrics(yobs[split['test']],ypred[split['test']],metrics) 
            models[type]['testmetrics'] = tstmetrics
            models[type]['pred'] = ypred


    return models




def TrainModels(df, drvvr, vr, modeltypes,split,metrics=None):

    if metrics is None:
        metrics = ['r2','bias','rmse','pr2']

    # initialising ML models
    models = dict()
    for type in modeltypes:
        if (type != 'MDS') & (type != 'MDSss'):
            models[type] = dict()
            for i in range(len(split['trainval'])):
                mod = MLModel(type)
                models[type][i] = mod
            # ensemble model of particular kind
            models[type]['ensemble'] = dict()
        else:
            models[type] = dict()
    # ensemble model of different models
    models['ensemble'] = dict()



    for type in modeltypes:    

        if (type != 'MDS') & (type != 'MDSss'):
        
            # tuning model hyperparameters using all the training/validation data
            # the same hyperparameters for all the models of the same type
            trainall = ~split['test'] & ~df[vr].isna()

            modhp = OptimizeHyperparameters(df[trainall], drvvr, vr, models[type][0])
            # modhp = dict()
            # if type == 'RF':
            #     modhp['params'] = dict()
            #     modhp['params']['n_estimators'] = 200
            #     modhp['params']['min_samples_split'] = 5
            #     modhp['params']['min_samples_leaf'] = 2
            #     modhp['params']['max_features'] = 0.33
            #     modhp['params']['bootstrap'] = True
            # elif type=='XGB':
            #     modhp['params'] = dict()
            #     modhp['params']['subsample'] = 0.75
            #     modhp['params']['max_depth'] = 10
            #     modhp['params']['min_child_weight'] = 5
            #     modhp['params']['colsample_bytree'] = 0.6



            for key in models[type].keys():
                models[type][key]['params'] = modhp['params']
                if key != 'ensemble':
                    models[type][key]['model'].set_params(**modhp['params'])


            keys = list(models[type].keys())
            for i in range(len(keys)):
                key = keys[i]
                mod = models[type][key]

                if key != 'ensemble':

                    if mod['params'] is None:
                        # tuning model hyperparameters individually for each model using training data
                        mod = OptimizeHyperparameters(df[split['trainval'][i][0]], drvvr, vr, mod)
                    else:
                        # model hyperparameters have been already defined
                        mod = MLfit(mod,df[split['trainval'][i][0]],drvvr,vr)

                    # performance metrics calculated agaist validation data
                    ypred = MLpredict(mod['model'],df[split['trainval'][i][1]],drvvr,vr)
                    valmetrics = CalcMetrics(df.loc[split['trainval'][i][1],vr],ypred,metrics)  

                    # performance metrics calculated against training data
                    ypred = MLpredict(mod['model'],df[split['trainval'][i][0]],drvvr,vr)
                    trainmetrics = CalcMetrics(df.loc[split['trainval'][i][0],vr],ypred,metrics)  

                    mod['valmetrics'] = valmetrics
                    mod['trainmetrics'] = trainmetrics

                    models[type][key] = mod

    return models




def MLModel(modeltype):

    model = dict()

    types = ['RF','XGB','kNN']
    if modeltype not in types:        
        msg = 'Model type %s not implemented.'%(modeltype,)
        logger.warning(msg)
        
    if modeltype=='RF':
        model['model'] = RandomForestRegressor()
        model['params_dist'] = dict()
        # number of trees in the forest
        model['params_dist']['n_estimators'] = [100,150,200,250,300,350,400]
        # amount of features used whe looking for the best split
        model['params_dist']['max_features'] = [1.0,0.33,0.5,'sqrt']
        # The minimum number of samples required to split an internal node
        model['params_dist']['min_samples_split'] = [2,5,10]
        # The minimum number of samples required to be at a leaf node
        model['params_dist']['min_samples_leaf'] = [1,2,4]
        # Whether bootstrap samples are used when building trees
        model['params_dist']['bootstrap'] = [True,False]

        # model['params'] = None
    if modeltype=='XGB':
        model['model'] = xgb.XGBRegressor(objective='reg:squarederror')
        model['params_dist'] = dict()
        # Subsample ratio of the training instance
        model['params_dist']['subsample'] = [0.5, 0.75, 1]
        # Maximum tree depth for base learners
        model['params_dist']['max_depth'] = [3, 5, 10, 15]
        # Minimum sum of instance weight(hessian) needed in a child
        model['params_dist']['min_child_weight'] = [2, 5, 10]
        # Subsample ratio of columns when constructing each tree
        model['params_dist']['colsample_bytree'] = [0.4, 0.6, 0.8, 1]


    return model


def FeatureSelection(df, drvvr, vr,method=None,tol=None):

    

    if method is None:
        method='RF'
    if tol is None:
        tol = 0.05

    # temporarily dropping time variables when selecting predictors since they likely dominate
    # this way we get physically meaningful predictors
    drvvrtmp = drvvr.copy()
    tvrs = ['ann_cycle_1','ann_cycle_2','daily_cycle_1','daily_cycle_2']
    for tvr in tvrs:
        drvvrtmp.remove(tvr)


    X, y, id = org_data(df, drvvrtmp, vr)
    
    vrsout = list()
    impout = dict()
    if method=='RF':
        model = RandomForestRegressor(n_estimators=200,max_features=0.33)

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = model.fit(X_train,y_train)
        r2_train = model.score(X_train,y_train)
        r2_test = model.score(X_test,y_test)

        # randomly permuting one feature at a time and calculating model performance score against test data
        # repeated 10 times
        # decrease in performance score tells the feature importance
        imp = permutation_importance(model,X_test,y_test,scoring='r2', n_repeats=10)
        vrs = X.columns[np.argsort(imp.importances_mean)[::-1]]
        impmean = imp.importances_mean[np.argsort(imp.importances_mean)[::-1]]
        impstd = imp.importances_std[np.argsort(imp.importances_mean)[::-1]]
        vrsout = list(vrs[impmean>tol])
        impout['mean'] = list(impmean)
        impout['std'] = list(impstd)
        impout['vrs'] = list(vrs)

        if ('wind_dir_cos' in vrsout) & ('wind_dir_sin' not in vrsout):
            vrsout.append('wind_dir_sin')
        elif ('wind_dir_sin' in vrsout) & ('wind_dir_cos' not in vrsout):
            vrsout.append('wind_dir_cos')

    elif method=='RFE':
        splitter = KFold(n_splits=3)

        model = RandomForestRegressor(n_estimators=200,max_features=0.33)
        selector = RFECV(
            PermImp(model,  scoring='r2', n_iter=10, cv=splitter),
            cv=splitter,
            scoring='r2',
            step=1
        )
        selector = selector.fit(X, y)

        # print(selector.ranking_)
        
        impout['mean'] = list(selector.cv_results_['mean_test_score'])
        impout['std'] = list(selector.cv_results_['std_test_score'])
        impout['vrs'] = list(selector.feature_names_in_)
        vrsout = list(selector.feature_names_in_)
        best_score = max(selector.cv_results_['mean_test_score'])
        best_score_std = selector.cv_results_['std_test_score'][best_score==selector.cv_results_['mean_test_score']]

        # selecting only those predictors that significantly increase the predictive power of the model
        indx = np.inf
        for i in range(len(impout['mean'])):
            if (impout['mean'][i]>best_score-best_score_std) & (i<indx):
                indx = i
        # at minimum two predictors
        if indx<1:
            indx=1

        if indx<len(impout['mean']):
            vrsout = vrsout[0:indx+1]
            



        if ('wind_dir_cos' in vrsout) & ('wind_dir_sin' not in vrsout):
            vrsout.append('wind_dir_sin')
        elif ('wind_dir_sin' in vrsout) & ('wind_dir_cos' not in vrsout):
            vrsout.append('wind_dir_cos')

    # adding tvars
    for tvr in tvrs:
        vrsout.append(tvr)
        impout['mean'].append(np.nan)
        impout['std'].append(np.nan)


    return vrsout,impout

    
def CalcMetrics(y,y_hat,metrics):


    out = dict()
    if (len(y)>1) & (len(y_hat)>1):
        for metric in metrics:
            if metric=='r2':
                out[metric] = sklearn.metrics.r2_score(y,y_hat)
            elif metric=='mae':
                out[metric] = sklearn.metrics.mean_absolute_error(y,y_hat)
            elif metric=='rmse':
                out[metric] = (sklearn.metrics.mean_squared_error(y,y_hat))**0.5
            elif metric=='mse':
                out[metric] = (sklearn.metrics.mean_squared_error(y,y_hat))
            elif metric=='pr2':
                out[metric] = (pearsonr(y,y_hat)[0])**2
            elif metric=='bias':
                out[metric] = np.mean(y_hat-y)
            else:
                msg = 'Model performance metric %s not implemented. Returning NaN.'%(metric,)
                logger.warning(msg)
                out[metric] = np.nan
    else:
        for metric in metrics:
            out[metric] = np.nan

    return out

def MLfit(mod,df,drvvr,vr):
    X, y, id = org_data(df,drvvr,vr)
    mod['model'].fit(X,y)

    return mod

def MLpredict(mod,df,drvvr,vr=None):
    
    df1 = df.copy()
    if vr is None:
        vr = 'flux'
        df1[vr] = 1

    y_out = pd.Series(index=df1.index,name=vr,dtype='float64')

    if len(df)>0:
        X, y, id = org_data(df1,drvvr,vr)
        y_hat = mod.predict(X)
        y_out[id] = y_hat

    return y_out

def AddTimeVariables(df,lat,lon,harvestdays=None):
    """
    Adds time variables to data
    """
    t = df['time']
    ts = t[0]
    te = t[-1]
    deltat = t[1]-t[0]
    
    SWIN_POT = tools.ClearSkyRadiation(lat,lon,ts,te,deltat)
    SWIN_POT['time'] = SWIN_POT['time'].dt.tz_localize(None)
    t2 = SWIN_POT['time']
    SWIN_POT = SWIN_POT.set_index(keys='time')
    SWIN_POT = SWIN_POT['SWIN_POT']

    dt2 = dt.timedelta(days=1)
    SWIN_POT_dmax = SWIN_POT.rolling(dt2,center=True).max()
    SWIN_POT_dmean = SWIN_POT_dmax.rolling(dt2,center=True).mean()
    SWIN_POT_grad = pd.Series(data=np.gradient(SWIN_POT_dmean),index=t2)
    SWIN_POT_grad = SWIN_POT_grad.rolling(dt2,center=True).mean()


    ann_cycle_1 = SWIN_POT_dmean.resample('30min').mean().reindex(index=t)
    ann_cycle_1 = (ann_cycle_1-ann_cycle_1.min())
    ann_cycle_1 = ann_cycle_1/ann_cycle_1.max()
    ann_cycle_2 = SWIN_POT_grad.resample('30min').mean().reindex(index=t)
    ann_cycle_2 = (ann_cycle_2-ann_cycle_2.min())
    ann_cycle_2 = ann_cycle_2/ann_cycle_2.max()
    daily_cycle_1 = SWIN_POT/SWIN_POT_dmax
    daily_cycle_2 = pd.Series(data=np.gradient(daily_cycle_1),index=t2)
    tmp = daily_cycle_2.rolling(dt2,center=True).max()
    tmp2 = daily_cycle_2.rolling(dt2,center=True).min()
    daily_cycle_2 = daily_cycle_2/(tmp-tmp2)+0.5
    daily_cycle_1 = daily_cycle_1.resample('30min').mean().reindex(index=t)
    daily_cycle_2 = daily_cycle_2.resample('30min').mean().reindex(index=t)

    # initialising days_since_harvest with a large number (three months)
    days_since_harvest = pd.Series(index=df.index,data=12*7)
    doy = (t-t[0])/dt.timedelta(days=1)

    if harvestdays is not None:
        harvestdays.append(t[-1].to_pydatetime())
        for i in range(len(harvestdays)-1):
            hd = harvestdays[i]
            hdn = harvestdays[i+1]
            tmp = (t-hd).abs()
            hddoy = doy[tmp==tmp.min()][0]
            days_since_harvest[(days_since_harvest.index>=hd) & (days_since_harvest.index<hdn)] = doy[(days_since_harvest.index>=hd) & (days_since_harvest.index<hdn)]-hddoy
            

    

    df['ann_cycle_1'] = ann_cycle_1
    df['ann_cycle_2'] = ann_cycle_2
    df['daily_cycle_1'] = daily_cycle_1
    df['daily_cycle_2'] = daily_cycle_2
    if harvestdays is not None:
        df['days_since_harvest'] = days_since_harvest


    # df['Month'] = df.index.month
    # df['Hour'] = df.index.hour+df.index.minute/60
    # df['Month_sin'] = np.sin((df.Month-1)*(2.*np.pi/12))
    # df['Month_cos'] = np.cos((df.Month-1)*(2.*np.pi/12))
    # df['Hour_sin'] = np.sin(df.Hour*(2.*np.pi/24))
    # df['Hour_cos'] = np.cos(df.Hour*(2.*np.pi/24))
    # df['Time'] = np.arange(0, len(df), 1)

    return df


def org_data(df, drvvr, vr):
    """
    Shuffles data, returns X, y and their indices
    """
    df = df[df[vr].notnull()]
    X, y, id = shuffle(df[drvvr], df[vr], df.index)

    return X, y, id

def OptimizeHyperparameters(df, drvvr, vr, model):
    """
    Optimizes hyperparameters using 5-fold cross validation and random search
    Returns best parameters
    """
    X, y, _ = org_data(df, drvvr, vr)

    mod = model['model']
    mod_rndm = RandomizedSearchCV(
            estimator=mod,
            param_distributions=model['params_dist'],
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1)

    mod_rndm.fit(X, y)
    model['model'] = mod_rndm.best_estimator_
    model['params'] = mod_rndm.best_params_

    return model


def cv_preds(df, vr,vrgf,drvvr, hyperparams, folds):
    """
    Gapfills y_col using x_cols as predictors
    """
    colsample_bytree=hyperparams.get('colsample_bytree')
    max_depth=hyperparams.get('max_depth')
    min_child_weight=hyperparams.get('min_child_weight')
    subsample=hyperparams.get('subsample')

    # Organize training data
    X, y, id = org_data(df, drvvr, vr)

    # Fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=colsample_bytree, max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample)

    # CV predict all y where measured data
    y_pred = cross_val_predict(model, X, y, cv=folds)
    df.loc[id,vrgf + '_cv'] = y_pred

    # Use all available measured data to predict real gaps
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=colsample_bytree, max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample).fit(X, y)
    y_pred_gaps = model.predict(np.asarray(df.loc[df[vrgf].isnull(), drvvr]))

    df.loc[df[vrgf].isnull(), vrgf] = y_pred_gaps

    return df




def GapDistribution(dat):

    # gap lengths
    gaps = dat.isnull().astype(int)
    observed = dat.notnull().astype(int)
    gap_lengths_with_zero = gaps.groupby(observed.cumsum()).sum()
    gap_lengths = gap_lengths_with_zero[gap_lengths_with_zero != 0].to_numpy()


    # gap_list = gap_lengths[gap_lengths < np.quantile(gap_lengths, 0.99)]

    gap_list = gap_lengths

    # convert list to normalized distribution
    gap_list = np.array(gap_list)
    gap_hist = np.bincount(gap_list)
    # gap probability mass function
    gap_pmf = gap_hist / np.sum(gap_hist)

    return gap_pmf



def sample_artificial_gaps(flux_data,
                           sampling_pmf,
                           eval_frac=0.1,
                           overlap_retries=20,
                           seed=1000,prev_gap=None):
    """ Randomly introduce gaps in a time series where the length of gaps
    are i.i.d. samples from a specified probability mass function.
    In order to keep the distribution of gap lengths equal to the sampling
    density, gaps are sampled in a non-overlapping way. This is first done by
    randomly selecting a starting location and a gap length. If an overlap
    occurs, a new starting location is selected. If a valid starting location
    cannot be found within a certain number of trials, the gap length is
    re-selected. This adds a small potential bias toward smaller gaps in return
    for avoiding infinite failure.
    Args:
        flux_data (np.array): time series to mask. Ignores existing gaps.
        sampling_pmf (np.array): Mass function/histogram with support [0, n]
        eval_frac (float): Percentage of non-NaN entries in the time series
            to mask out. This is done by introducing gaps until the threshold
            is reached.
        overlap_retries (int): Number of trials of adding a gap before
            re-selecting gap length.
        seed (int): numpy random seed to fix sampling
    Return:
        masked_series (np.array): time series with gaps represented as NaNs
    """
    # adopted from https://github.com/stanfordmlgroup/methane-gapfill-ml
    # Irvin, J., Zhou, S., McNicol, G., Lu, F., Liu, V., Fluet-Chouinard, E., ... 
    # & Jackson, R. B. (2021). Gap-filling eddy covariance methane fluxes: Comparison 
    # of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands. 
    # Agricultural and Forest Meteorology, 308, 108528.


    np.random.seed(seed)
    observed = np.isfinite(flux_data)
    if prev_gap is None:
        prev_gap = pd.Series(data=False,index=flux_data.index)
    observed = np.isfinite(flux_data) & ~prev_gap

    gap_mask = ~prev_gap.copy()

    prop_masked = 0.
    while prop_masked < eval_frac:
        # pick random index and gap length
        trials = 0
        rand_idx = np.random.choice(np.where(gap_mask)[0])
        rand_gap = np.random.choice(len(sampling_pmf), p=sampling_pmf)

        if flux_data.index.inferred_type=='datetime64':
            rand_idxt = gap_mask.index[rand_idx]
            rand_gapt = dt.timedelta(minutes=30)*(rand_gap-1)
            gplnght = len(gap_mask[rand_idxt:rand_idxt+rand_gapt])
        else:
            gplnght = len(gap_mask[rand_idx:rand_idx+rand_gap])

        # retry if overlaps with previously chosen gaps
        while (np.any(~gap_mask[rand_idx:rand_idx + rand_gap])) & (gplnght<rand_gap):
            rand_idx = np.random.choice(np.where(gap_mask)[0])
            trials += 1
            if trials > overlap_retries:
                rand_gap = np.random.choice(len(sampling_pmf), p=sampling_pmf)



            if flux_data.index.inferred_type=='datetime64':
                rand_idxt = gap_mask.index[rand_idx]
                rand_gapt = dt.timedelta(minutes=30)*(rand_gap-1)
                gplnght = len(gap_mask[rand_idxt:rand_idxt+rand_gapt])
            else:
                gplnght = len(gap_mask[rand_idx:rand_idx+rand_gap])
            
            

        # gap successfully added
        if flux_data.index.inferred_type=='datetime64':
            gap_mask[rand_idxt:rand_idxt+rand_gapt] = False
        else:
            gap_mask[rand_idx:rand_idx + rand_gap] = False

        # recompute the total observed percentage masked
        prop_masked = np.sum(observed * ((~gap_mask) & (~prev_gap))) / np.sum(observed)


    masked_series = flux_data.copy()
    masked_series[(~gap_mask) & (~prev_gap)] = np.nan


    return masked_series    


def MDS(df,vr,vrgf,drvvr,tol,Nmin=None,vrunc=None,timeisna=None,method=None,subsample=None):

    if Nmin is None:
        Nmin=2
    if Nmin is not round(Nmin):
        msg = 'MDS algorithm accepts only integer number for minimum amount of data. Rounding to integer.'
        logger.warning(msg)
        Nmin=round(Nmin)
    if Nmin<2:
        msg = 'Nmin must be larger than 1. Forcing to 2.'
        logger.warning(msg)
        Nmin=2
    if method is None:
        method = 'mean'
    if subsample is None:
        subsample = dict()
        for drv in drvvr:
            subsample[drv] = False

    # periods with gaps that are filled. If None, then filling all the gaps
    if timeisna is None:
        timeisna = df.loc[df[vrgf].isnull(),'time']


    # MDS algorithm
    # step 1: Look-up table with window size +/- 7 days
    df = LUT(df,vr,vrgf,drvvr,tol,deltadays=7,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method,subsample=subsample)
    intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
    timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    # df[vrgf+'_1'] = df[vrgf]
    # step 2: Look-up table with window size +/- 14 days
    df = LUT(df,vr,vrgf,drvvr,tol,deltadays=14,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method,subsample=subsample)
    intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
    timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    # df[vrgf+'_2'] = df[vrgf]
    # step 3: Look-up table with main driver only, window size +/- 7 days
    df = LUT(df,vr,vrgf,[drvvr[0]],tol,deltadays=7,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method,subsample=subsample)
    intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
    timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    # df[vrgf+'_3'] = df[vrgf]
    # step 4: Mean diurnal course with window size of 0 (same day)
    df = MDC(df,vr,vrgf,deltadays=0,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method)
    intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
    timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    # df[vrgf+'_4'] = df[vrgf]
    # step 5: Mean diurnal course with window size of +/-1 day and +/-2 days
    df = MDC(df,vr,vrgf,deltadays=1,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method)
    intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
    timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    df = MDC(df,vr,vrgf,deltadays=2,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method)
    intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
    timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    # df[vrgf+'_5'] = df[vrgf]
    # step 6: Look-up table with window size of +/- 21 to +/- 70 days
    for tdays in range(21,70,7):
        df = LUT(df,vr,vrgf,drvvr,tol,deltadays=tdays,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method,subsample=subsample)
        intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
        timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    # df[vrgf+'_6'] = df[vrgf]
    # step 7: Look-up table with main driver only, window size of +/- 14 to +/- 70 days
    for tdays in range(14,70,7):
        df = LUT(df,vr,vrgf,[drvvr[0]],tol,deltadays=tdays,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method,subsample=subsample)
        intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
        timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')
    # df[vrgf+'_7'] = df[vrgf]
    # step 8: Mean diurnal course with window size of +/- 7 to +/- 210 days
    for tdays in range(7,210,7):
        df = MDC(df,vr,vrgf,deltadays=tdays,Nmin=Nmin,vrunc=vrunc,timeisna=timeisna,method=method)
        intrsct =  list(set(timeisna) & set(df.loc[df[vrgf].isnull(),'time']))
        timeisna = pd.Series(index=intrsct,data=intrsct,name=timeisna.name,dtype='datetime64[ns]')


    return df


def LUT(df,vr,vrgf,drvvr,tol,deltadays=None,Nmin=None,vrunc=None,timeisna=None,method=None,subsample=None):

    dfout = df.copy()

    if Nmin is None:
        Nmin=2
    if Nmin is not round(Nmin):
        msg = 'LUT algorithm accepts only integer number for minimum amount of data. Rounding to integer.'
        logger.warning(msg)
        Nmin=round(Nmin)
    if Nmin<2:
        msg = 'Nmin must be larger than 1. Forcing to 2.'
        logger.warning(msg)
        Nmin=2

    if deltadays is None:
        deltadays = 1
    if deltadays is not round(deltadays):
        msg = 'LUT algorithm accepts only integer number of days. Rounding to integer.'
        logger.warning(msg)
        deltadays = round(deltadays)
    if deltadays<1:
        msg = 'deltadays must be larger than 1. Forcing to 1.'
        logger.warning(msg)
        deltadays = 1

    if method is None:
        method = 'mean'

        

    if subsample is None:
        subsample = dict()
        for drv in drvvr:
            subsample[drv] = False
            

    # periods with gaps that are filled. If None, then filling all the gaps
    if timeisna is None:
        timeisna = df.loc[df[vrgf].isnull(),'time']

    deltat = dt.timedelta(days=int(deltadays))

    df['bool'] = 1
    df.loc[df[vr].isnull(),'bool'] = 0
    df['group'] = ''

    # looping over the gaps
    for t in timeisna:
        dftmp = df.loc[df['time'].between(t-deltat,t+deltat),drvvr+[vr,'time','group','bool']].copy()
        

        grp = 'a'
        for drv in drvvr:

            # value for the driver at this particular time
            drvval = dftmp.loc[dftmp['time']==t,drv][0]
            
            # tolerance around drvval
            tolval = tol[drv][0][1]
            for i in range(len(tol[drv])):
                if (drvval>tol[drv][i][0][0]) & (drvval<=tol[drv][i][0][1]):
                    tolval = tol[drv][i][1]


            grp = chr(ord(grp)+1)
            if subsample[drv]:
                lms = [drvval-tolval,drvval]
                dftmp.loc[dftmp[drv].between(lms[0],lms[1]),'group'] = dftmp.loc[dftmp[drv].between(lms[0],lms[1]),'group']+grp


                lms = [drvval,drvval+tolval]
                grp1 = chr(ord(grp)+1)
                dftmp.loc[(dftmp[drv].between(lms[0],lms[1])) & (~dftmp['group'].str.contains(grp)),'group'] = dftmp.loc[dftmp[drv].between(lms[0],lms[1]) & (~dftmp['group'].str.contains(grp)),'group']+grp1
                grp = grp1
            
            else:
                lms = [drvval-tolval,drvval+tolval]
                dftmp.loc[dftmp[drv].between(lms[0],lms[1]),'group'] = dftmp.loc[dftmp[drv].between(lms[0],lms[1]),'group']+grp
        dftmp = dftmp.loc[dftmp['group'].str.len()>=len(drvvr)]

        # amount of non-NaN values
        ndat = dftmp['bool'].sum()

        if ndat>=Nmin:
            # enough data for averaging            
            if method=='mean':
                dfout.loc[dfout['time']==t,vrgf] = dftmp.groupby('group').mean(numeric_only=True).mean(numeric_only=True)[vr]
            if method=='median':
                dfout.loc[dfout['time']==t,vrgf] = dftmp.groupby('group').median(numeric_only=True).median(numeric_only=True)[vr]
            if vrunc is not None:
                dfout.loc[dfout['time']==t,vrunc] = dftmp.groupby('group').std().mean(numeric_only=True)[vr]



    return dfout


def MDCold(df,vr,vrgf,deltadays=None,method=None,Nmin=None,vrunc=None,timeisna=None):

    dfout = df.copy()

    if Nmin is None:
        Nmin=2
    if Nmin is not round(Nmin):
        msg = 'MDC algorithm accepts only integer number for minimum amount of data. Rounding to integer.'
        logger.warning(msg)
        Nmin=round(Nmin)
    if Nmin<2:
        msg = 'Nmin must be larger than 1. Forcing to 2.'
        logger.warning(msg)
        Nmin=2

    if deltadays is None:
        deltadays = 1
    if deltadays is not round(deltadays):
        msg = 'MDC algorithm accepts only integer number of days. Rounding to integer.'
        logger.warning(msg)
        deltadays = round(deltadays)
    if deltadays<0:
        msg = 'deltadays must be larger than 0. Forcing to 0.'
        logger.warning(msg)
        deltadays = 0

    if method is None:
        method = 'mean'
    if method!='mean' and method!='median':
        msg = 'Unknown method given (%s) for calculating diurnal patterns. Using ''mean'' instead.'%(method,)
        logger.warning(msg)
        method = 'mean'

        
    deltat = dt.timedelta(days=int(deltadays))

    
    # periods with gaps that are filled. If None, then filling all the gaps
    if timeisna is None:
        timeisna = df.loc[df[vrgf].isnull(),'time']


    df['bool'] = 1
    df.loc[df[vr].isnull(),'bool'] = 0

    for t in timeisna:
        
        if deltadays==0:
            mndt = dt.datetime(t.date().year,t.date().month,t.date().day)
            mxdt = dt.datetime(t.date().year,t.date().month,t.date().day)+dt.timedelta(days=1)
        else:
            mndt = dt.datetime(t.date().year,t.date().month,t.date().day)-deltat
            mxdt = dt.datetime(t.date().year,t.date().month,t.date().day)+deltat+dt.timedelta(days=1)


        dftmp = pd.DataFrame(df.loc[(df['time']>=mndt) & (df['time']<=mxdt) & (df[vr].notnull()),[vr,'bool']])
        dftmp['hour'] = dftmp.index.hour  
        


        if method=='mean':
            val = dftmp.loc[dftmp['hour']==t.hour,vr].mean()
        elif method=='median':
            val = dftmp.loc[dftmp['hour']==t.hour,vr].median()

        if vrunc is not None:
            uncval = dftmp.loc[dftmp['hour']==t.hour,vr].std()

        ndat = dftmp.loc[dftmp['hour']==t.hour,'bool'].sum()
        
        if ndat>=Nmin:
            dfout.loc[dfout['time']==t,vrgf] = val
            if vrunc is not None:
                dfout.loc[dfout['time']==t,vrunc] = uncval



    return dfout


def MDC(df,vr,vrgf,deltadays=None,method=None,Nmin=None,vrunc=None,timeisna=None,deltat=None):

    dfout = df.copy()

    z4 = tools.DiurnalPattern(df[vr],df['time'],method=method,deltadays=deltadays,Nmin=Nmin,deltat=deltat)
    if vrunc is not None:
        z5 = tools.DiurnalPattern(df[vr],df['time'],method='std',deltadays=deltadays,Nmin=Nmin,deltat=deltat)



    # periods with gaps that are filled. If None, then filling all the gaps
    if timeisna is None:
        timeisna = df.loc[df[vrgf].isnull(),'time']
    
    for t in timeisna:
        dfout.loc[t,vrgf] = z4[t]
        if vrunc is not None:
            dfout.loc[t,vrunc] = z5[t]

    return dfout

def InterpolateShortGaps(df,vr,N,method=None):
    """Filling short gaps within the time series with interpolation.
    Args:
        df (DataFrame): data
        vr (str): column in df to be filled
        N (int): max length for the gaps to be filled
        method (str, optional): method used for interpolation. Defaults to None.
    Returns:
        DataFrame: dataframe where the short gaps in column are filled with selected interpolation method
    """


    if method is None:
        method = 'linear'

    if (method!='linear') & (method!='nearest') & (method!='quadratic') & (method!='cubic') & (method!='spline'):
        msg = 'Unknown method given (%s) for interpolating short gaps. Using ''linear'' instead.'%(method,)
        logger.warning(msg)
        method = 'linear'
    
    dat = df[vr]

    dat_interpolated = dat.interpolate(method=method)

    mask = dat.isna()
    x = (mask.groupby((mask != mask.shift()).cumsum()).transform(lambda x: len(x) > N)*mask)
    dat_interpolated = dat_interpolated.loc[~x]

    df[vr] = dat_interpolated

    return df


def GapfillBiomet(df,vrs,config=None,ctype=None):
    """
    df - dataframe
    vrs - variables
    
    """
    cols = df.columns

    # making sure that certain variables are gapfilled in a specific order
    if 'Tsurf' in vrs:
        vrs.insert(0, vrs.pop(vrs.index('Tsurf')))
    if 'TA_1_1_1' in vrs:
        vrs.insert(0, vrs.pop(vrs.index('TA_1_1_1')))
    if 'PA_1_1_1' in vrs:
        vrs.insert(0, vrs.pop(vrs.index('PA_1_1_1')))
    if 'VPD' in vrs:
        vrs.append(vrs.pop(vrs.index('VPD')))
    if 'TS_1_1_1' in vrs:
        vrs.append(vrs.pop(vrs.index('TS_1_1_1')))
    if 'TS_2_1_1' in vrs:
        vrs.append(vrs.pop(vrs.index('TS_2_1_1')))
    if 'TS_3_1_1' in vrs:
        vrs.append(vrs.pop(vrs.index('TS_3_1_1')))


    for vr in vrs:
        if vr not in cols:        
            msg = 'Variable (%s) not in the data set. Skipping the variable.'%(vr,)
            logger.warning(msg)
        elif (df[vr].isna().sum()>0.8*len(df)):
            msg = 'Too much gaps in %s (over 80 %%). Not gapfilling the variable.'%(vr,)
            logger.warning(msg)

        else:
            vrgf = vr+'_gf'
            
            if 'PPFD_1_1_1'==vr:
                # Photosynthetic Photon Flux Density
                df[vrgf] = df['PPFD_1_1_1']
                if 'BF5_PARtot_1_1_1' in cols:
                    df.loc[df[vrgf].isna(),vrgf] = df.loc[df[vrgf].isna(),'BF5_PARtot_1_1_1']
                if 'SWIN_1_1_1' in cols:
                    df.loc[df[vrgf].isna(),vrgf] = df.loc[df[vrgf].isna(),'SWIN_1_1_1']/0.5
                df = InterpolateShortGaps(df,vrgf,4)

                df.loc[df[vrgf].isna(),vrgf] = df.loc[df[vrgf].isna(),'SWIN_POT']/0.5

            
            elif 'SWIN_1_1_1'==vr:
                # Incoming short wave radiation
                df[vrgf] = df['SWIN_1_1_1']
                if 'PPFD_1_1_1' in cols:
                    df.loc[df[vrgf].isna(),vrgf] = df.loc[df[vrgf].isna(),'PPFD_1_1_1']*0.5
                if 'BF5_PARtot_1_1_1' in cols:
                    df.loc[df[vrgf].isna(),vrgf] = df.loc[df[vrgf].isna(),'BF5_PARtot_1_1_1']*0.5
                df = InterpolateShortGaps(df,vrgf,4)
                if 'ALB_1_1_1' in cols:
                    # Using weekly median albedo and SWOUT to fill SWIN gaps
                    ALB = df['ALB_1_1_1'].rolling('7D').median()
                    ALB[ALB>1] = 1
                    ALB[ALB<0.1] = 0.1
                    df.loc[df[vrgf].isna(),vrgf] = df.loc[df[vrgf].isna(),'SWOUT_1_1_1']/ALB[df[vrgf].isna()]
                    
                
                df.loc[df[vrgf].isna(),vrgf] = df.loc[df[vrgf].isna(),'SWIN_POT']


            
            elif 'TA_1_1_1'==vr:
                # air temperature
                df[vrgf] = df['TA_1_1_1']
                df = InterpolateShortGaps(df,vrgf,4)
                df = MDC(df,'TA_1_1_1',vrgf,method='median',deltadays=1,Nmin=3)
                df = MDC(df,'TA_1_1_1',vrgf,method='median',deltadays=5,Nmin=3)
                df = MDC(df,'TA_1_1_1',vrgf,method='median',deltadays=10,Nmin=3)
                df = MDC(df,'TA_1_1_1',vrgf,method='median',deltadays=30,Nmin=3)

                

            
            elif 'P_RAIN_1_1_1'==vr:
                # precipitation
                df[vrgf] = df['P_RAIN_1_1_1']
                df[vrgf] = df[vrgf].fillna(value=0)


            
            elif 'RH_1_1_1'==vr:
                # relative humidity
                df[vrgf] = df['RH_1_1_1']
                
                if ('h2o_mixing_ratio' in cols) & ('TA_1_1_1_gf' in df.columns) & ('PA_1_1_1_gf' in df.columns):
                    df['h2o_mixing_ratio_gf'] = df['h2o_mixing_ratio']
                    df = InterpolateShortGaps(df,'h2o_mixing_ratio_gf',4)
                    RH2 = tools.CalcRH(df['h2o_mixing_ratio_gf'],df['TA_1_1_1_gf'],df['PA_1_1_1_gf'])
                    df.loc[df[vrgf].isna(),vrgf] = RH2.loc[df[vrgf].isna()]
                df = InterpolateShortGaps(df,vrgf,4)
                df = MDC(df,'RH_1_1_1',vrgf,method='median',deltadays=1)
                df = MDC(df,'RH_1_1_1',vrgf,method='median',deltadays=5)
                df = MDC(df,'RH_1_1_1',vrgf,method='median',deltadays=10)
                df = MDC(df,'RH_1_1_1',vrgf,method='median',deltadays=30)
                df.loc[df[vrgf]>100,vrgf] = 100


            
            elif 'VPD'==vr:
                # vapor pressure deficit
                df[vrgf] = df['VPD']
                if ('RH_1_1_1_gf' in df.columns) &  ('TA_1_1_1_gf' in df.columns):
                    VPD2 = tools.CalcVPD(df['RH_1_1_1_gf'],df['TA_1_1_1_gf'])
                    df.loc[df[vrgf].isna(),vrgf] = VPD2.loc[df[vrgf].isna()]
                df = InterpolateShortGaps(df,vrgf,4)

                df = MDC(df,'VPD',vrgf,method='median',deltadays=1)
                df = MDC(df,'VPD',vrgf,method='median',deltadays=5)
                df = MDC(df,'VPD',vrgf,method='median',deltadays=10)
                df = MDC(df,'VPD',vrgf,method='median',deltadays=30)

            elif 'SHF_' in vr:
                # soil heat flux
                SHFvrs = [vr]
                if ('SHF_2_1_1' in cols) & (vr!='SHF_2_1_1'):
                    SHFvrs.append('SHF_2_1_1')
                if ('SHF_1_1_1' in cols) & (vr!='SHF_1_1_1'):
                    SHFvrs.append('SHF_1_1_1')
                if ('SHF_3_1_1' in cols) & (vr!='SHF_3_1_1'):
                    SHFvrs.append('SHF_3_1_1')

                df['SHF_median'] = df[SHFvrs].median(axis=1)
                df[vrgf] = df['SHF_median']
                df = InterpolateShortGaps(df,vrgf,4)

                df = MDC(df,'SHF_median',vrgf,method='median',deltadays=1)
                df = MDC(df,'SHF_median',vrgf,method='median',deltadays=5)
                df = MDC(df,'SHF_median',vrgf,method='median',deltadays=10)
                df = MDC(df,'SHF_median',vrgf,method='median',deltadays=30)

                df = df.drop(labels='SHF_median',axis=1)

            
            elif 'TS_1_1_1'==vr:
                if 'TA_1_1_1_gf' in df.columns:
                    Ttop = df['TA_1_1_1_gf']
                elif 'Tsurf_gf' in df.columns:
                    Ttop = df['Tsurf_gf']
                else:
                    Ttop = None
                    
                df[vrgf] = df['TS_1_1_1']
                df = InterpolateShortGaps(df,vrgf,4)

                if config is not None:
                    z = config['biomet_info']['z'][0][vr]
                else:
                    z = None

                if Ttop is not None:
                    TSgf = GapfillSoilT(df[vrgf],Ttop,z=z,ctype=ctype)
                    TSgf[TSgf<0.1] = 0.1
                    df.loc[df[vrgf].isna(),vrgf] = TSgf.loc[df[vrgf].isna()]
            
            elif 'TS_2_1_1'==vr:
                if 'TA_1_1_1_gf' in df.columns:
                    Ttop = df['TA_1_1_1_gf']
                elif 'Tsurf_gf' in df.columns:
                    Ttop = df['Tsurf_gf']
                else:
                    Ttop = None
                    
                df[vrgf] = df['TS_2_1_1']
                df = InterpolateShortGaps(df,vrgf,4)

                if config is not None:
                    z = config['biomet_info']['z'][0][vr]
                else:
                    z = None

                if Ttop is not None:
                    TSgf = GapfillSoilT(df[vrgf],Ttop,z=z,ctype=ctype)
                    TSgf[TSgf<0.1] = 0.1
                    df.loc[df[vrgf].isna(),vrgf] = TSgf.loc[df[vrgf].isna()]

            
            elif 'TS_3_1_1'==vr:
                if 'TA_1_1_1_gf' in df.columns:
                    Ttop = df['TA_1_1_1_gf']
                elif 'Tsurf_gf' in df.columns:
                    Ttop = df['Tsurf_gf']
                else:
                    Ttop = None
                    
                df[vrgf] = df['TS_3_1_1']
                df = InterpolateShortGaps(df,vrgf,4)

                if config is not None:
                    z = config['biomet_info']['z'][0][vr]
                else:
                    z = None

                if Ttop is not None:
                    TSgf = GapfillSoilT(df[vrgf],Ttop,z=z,ctype=ctype)
                    TSgf[TSgf<0.1] = 0.1
                    df.loc[df[vrgf].isna(),vrgf] = TSgf.loc[df[vrgf].isna()]

            else:
                vrgf = vr+'_gf'
                df[vrgf] = df[vr]
                df = InterpolateShortGaps(df,vrgf,4)

                df = MDC(df,vr,vrgf,method='median',deltadays=1)
                df = MDC(df,vr,vrgf,method='median',deltadays=5)
                df = MDC(df,vr,vrgf,method='median',deltadays=10)
                df = MDC(df,vr,vrgf,method='median',deltadays=30)

    return df


def GapfillSoilT(TS,Ttop,z=None,ctype=None):


    TSgf = TS.copy()

    if ctype is None:
        ctype = 'emp'
    if (ctype=='theor') & (z is None):
        ctype = 'emp'
    if (ctype!='theor') & (ctype!='emp'):
        ctype = 'emp'


    # determining site specific time series of bulk heat conductivity
    if ctype=='emp':
        TS1 = TS.rolling('1h').mean() # trying to remove periods when dTdt is zero
        dTdt = np.gradient(TS1)
        dTdt[dTdt==0] = np.nan
        dTdz = (Ttop-TS1)

        
        coef = dTdt/dTdz   
        coef = coef.rolling('7D',min_periods=5).median()
        coef[coef<0] = np.nan 
        coef.name = 'coef'
        tmp = InterpolateShortGaps(pd.DataFrame(coef),'coef',48*30)
        coef = tmp['coef']
        coef[coef.isna()] = coef.median()

    elif ctype=='theor':
        KT = 0.65
        CS = 1.2e6
        CICE = 8.9e6
        CA = CS+CICE
        coef = pd.Series(index=TSgf.index,data=30*60*KT/(CA*(2*z)**2))

    coeforig = coef.copy()

    if np.isnan(TSgf[0]):
        TSgf[0] = Ttop[0]
    if TSgf[0]<0:
        TSgf[0] = 0.1

    # periods with consecutive missing values
    per = tools.NaNPeriods(TSgf)

    # looping over the data gaps
    for i in range(len(per)):
        ifrst = per.loc[i,'first']
        ilst = per.loc[i,'last']
        
        if (ifrst>TS.index[0]):
            loopind = 0
            dTS = 998
            dTSprev = 999
            indeces = TSgf.index[(TSgf.index>=ifrst) & (TSgf.index<=ilst)]
            while (loopind<10) & (abs(dTS)>0.2) & (abs(dTSprev)>abs(dTS)) & (np.mean(coeforig[indeces]/coef[indeces])<5) & (np.mean(coeforig[indeces]/coef[indeces])>0.2):
                dTSprev = dTS
                for j in indeces:

                    #previous index
                    if TSgf.index.inferred_type=='datetime64':
                        jp = j-1*TSgf.index.freq
                    else:
                        jp = j-1

                    # Eq. 12 in https://doi.org/10.5194/hess-8-706-2004
                    TSgf.loc[j] = (1-coef[j])*TSgf.loc[jp]+coef.loc[j]*Ttop.loc[jp]


                #next index
                if TSgf.index.inferred_type=='datetime64':
                    jn = ilst+1*TSgf.index.freq
                else:
                    jn = ilst+1
                dTS = TSgf[ilst]-TS[jn]
                # if loopind == 0:
                #     dTA = Ttop[indeces].mean()-TSgf[indeces].mean()
                    
                dTA = Ttop[indeces].mean()-TSgf[indeces].mean()
                if (abs(dTSprev)>abs(dTS)):
                    if (dTS>0) & (dTA>0):
                        # too much downward heat transfer
                        coef[indeces] = coef[indeces]*.7
                    if (dTS>0) & (dTA<0):
                        # too little downward heat transfer
                        coef[indeces] = coef[indeces]*1.3
                    if (dTS<0) & (dTA<0):
                        # too much downward heat transfer
                        coef[indeces] = coef[indeces]*.7
                    if (dTS<0) & (dTA>0):
                        # too little downward heat transfer
                        coef[indeces] = coef[indeces]*1.3
                loopind = loopind+1

            if (abs(dTSprev)<abs(dTS)):
                # if the latest results was worse than the one before that, then reverting to the previous result
                for j in indeces:

                    #previous index
                    if TSgf.index.inferred_type=='datetime64':
                        jp = j-1*TSgf.index.freq
                    else:
                        jp = j-1

                    # Eq. 12 in https://doi.org/10.5194/hess-8-706-2004
                    TSgf.loc[j] = (1-coef[j])*TSgf.loc[jp]+coef.loc[j]*Ttop.loc[jp]
    return TSgf

def AddLaggedVariables(df,vrs,deltats=None):

    dfout = df.copy()
    if deltats is None:
        deltats = [3/24,1,7,14]
    for vr in vrs:
        for deltat in deltats:
            if deltat>=1:
                vrout = vr+'_'+f"{deltat:.0f}" + '_days'
            else:
                vrout = vr+'_'+f"{deltat*24:.0f}" + '_hours'
            if vr in dfout.columns:
                    dt2 = dt.timedelta(days=deltat)
                    tmp = dfout[vr].rolling(dt2).mean()
                    tmp = tmp.fillna(value=tmp.median())
                    dfout[vrout] = tmp
            else:
                dfout[vrout] = pd.Series(index=df.index)
    return dfout

def ConvertWindDir(df,vr):
    dfout = df.copy()
    WD = dfout[vr] / 180 * np.pi
    notna = ~WD.isna()
    dfout.loc[notna, vr+'_sin'] = np.sin(WD[notna])
    dfout.loc[notna, vr+'_cos'] = np.cos(WD[notna])
    
    return dfout

