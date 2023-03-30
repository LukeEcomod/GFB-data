# -*- coding: utf-8 -*-
"""
.. module: plottools
    :synopsis: Plotting flux and timeseries data.
.. moduleauthor:: Olli Peltola & Samuli Launiainen

Based on Olli's PyECPC_utilities/plottools.py

Last edit 21.3.2023 / Samuli

"""

from ast import Index
from bokeh.core.property.color import Alpha
from bokeh.models.tools import HoverTool
from bokeh.models import Slope,DateRangeSlider,ColumnDataSource,ColorBar,BasicTicker,DatetimeTickFormatter,VArea
from bokeh.layouts import column
import numpy as np
import numpy.matlib
import datetime as dt
#import os,sys,glob
#import time
#import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.io import save
from cmcrameri import cm
#import xarray as xr
import logging
from pyproj import Transformer
#from skimage import measure
#import cartopy.crs as ccrs
import pandas as pd
import scipy.odr as odr
#import scipy.stats as stats
import panel as pn
#from bokeh.resources import INLINE
#import shapely.geometry as sgeom
#import geopandas as gpd
import holoviews as hv

hv.extension('bokeh')
pn.extension()

logger = logging.getLogger("ecpc_log")


def PlotTimeSeries(time,dat,title='',ylb='',xlb='',flout='',saveshowflag=2,ylm=[],xlm=[],plottype='line',p=None,size=4):

    TOOLS = "save,pan,box_zoom,reset,wheel_zoom"

    # defining x-axis limits
    if len(xlm)<2:
        te = dt.datetime.now()
        ts = te-dt.timedelta(days=20)
        xlm = [ts,te]

    # defining y-axis limits using the data between xlm
    if len(ylm)<2:
        yb = []
        yt = []
        if len(dat)>0:
            for key in dat.keys():
                    MAD = np.nanmedian(np.abs(dat[key][time[key].between(xlm[0],xlm[1])]-np.nanmedian(dat[key][time[key].between(xlm[0],xlm[1])])))
                    yb.append(np.nanpercentile(dat[key][time[key].between(xlm[0],xlm[1])],1)-MAD*0.5)
                    yt.append(np.nanpercentile(dat[key][time[key].between(xlm[0],xlm[1])],99)+MAD*0.5)
            ylm = [np.nanmin(yb),np.nanmax(yt)]
            if np.isnan(ylm[0]):
                ylm[0] = np.nanmin([0,ylm[1]])
            if np.isnan(ylm[1]):
                ylm[1] = np.nanmax([1,ylm[0]])
        else:
            ylm = [0,1]
            

    clrindx = np.round(np.linspace(0,255,len(dat)))

    if p is None:
        p = figure(title=title,x_axis_type="datetime", x_axis_label=xlb, y_axis_label=ylb,
        x_range=xlm,y_range=ylm,aspect_ratio =2.5,width = 800,tools=TOOLS, toolbar_location='above')
        p.add_tools(HoverTool(
        tooltips=[
            ('Time', '@x{%F %T}'),
            ('Value','@y')
        ],
        formatters={
            '@x': 'datetime',
        }
        ))  
        # removing grid lines
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.outline_line_color = "black"
        p.outline_line_width = 1
        # adding zero line
        p.line([np.min([xlm[0],dt.datetime.now()-dt.timedelta(days=60)]),dt.datetime.now()], [0,0], line_width=1,line_color='grey',line_dash='dashed')

    # add a line renderer with legend and line thickness to the plot
    keys = list(dat.keys())
    for i in range(len(keys)):
        key = keys[i]
        # clr = list(cm.batlow(int(clrindx[i,])))
        # clr = [int(np.round(i * 256)) for i in clr]
        clr = list(cm.batlowS(i,0.8,True))
        clr[3] = 0.8
        clr = tuple(clr)
        if plottype=='line':
            p.line(time[key], dat[key], legend_label=key, line_width=1.4,line_color=clr)
            # p.line(x='time', y='Value',source=source, legend_label=key, line_width=1,line_color=clr)
        elif plottype=='circle':
            # p.circle(time[key], dat[key], legend_label=key,line_color="black",fill_color=clr, size=12,fill_alpha=0.8)
            # p.scatter(time[key], dat[key], legend_label=key,line_color="black",fill_color=clr[0:3])      

            clrs = np.matlib.repmat(clr[0:3], len(dat[key]), 1)
            p.scatter(time[key], dat[key], legend_label=key,line_color="black",fill_color=clrs,size=size)

        elif plottype=='boxplot':
            dates = time[key].dt.date
            p = PlotBoxPlot(dates, dat[key],key,lclr='black',fclr=clr,p=p)
    # p.line(ncdat['LE'], legend_label="Temp.", line_width=2)

    p.legend.location = "top_left"
    
    # show the results
    if saveshowflag==1:
        # save
        if len(flout)>0:
            save(p,flout)
    elif saveshowflag==2:
        # show
        show(p)
    # elif saveshowflag==0:
    #     # only return
    
    return p


def PlotDielPattern(time,dat,title='',ylb='',xlb='',flout='',te=None,saveshowflag=2,ylm=[],plottype='line',p=None):

    TOOLS = "save,pan,wheel_zoom,reset"


    # previous day
    if te is not None:
        ts = dt.datetime.combine(te.date(), dt.datetime.min.time())
    else:
        te = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
        ts = te-dt.timedelta(days=1)


    # defining y-axis limits
    if len(ylm)<2:
        yb = []
        yt = []
        if len(dat)>0:
            MAD = np.nanmedian(np.abs(dat[time.between(ts,te)]-np.nanmedian(dat[time.between(ts,te)])))
            yb.append(np.nanpercentile(dat[time.between(ts,te)],1)-MAD*0.5)
            yt.append(np.nanpercentile(dat[time.between(ts,te)],99)+MAD*0.5)
            ylm = [np.nanmin(yb),np.nanmax(yt)]
            if np.isnan(ylm[0]):
                ylm[0] = np.nanmin([0,ylm[1]])
            if np.isnan(ylm[1]):
                ylm[1] = np.nanmax([1,ylm[0]])
        else:
            ylm = [0,1]

    if len(dat)>0:
        # calculating median diel pattern
        datmedian = dat.groupby(lambda x: x.hour).median()
        datmedian = datmedian.rename('median')
        dat25perc = dat.groupby(lambda x: x.hour).quantile(.25)
        dat25perc = dat25perc.rename('25perc')
        dat75perc = dat.groupby(lambda x: x.hour).quantile(.75)
        dat75perc = dat75perc.rename('75perc')

        datstat = pd.merge(datmedian, dat25perc, right_index = True,
               left_index = True)
        datstat = pd.merge(datstat, dat75perc, right_index = True,
               left_index = True)

        time2 = pd.date_range(start=ts, end=te, freq='1H')
        time2 = time2+dt.timedelta(minutes=30)
        datstat['time'] = pd.to_datetime(time2)
        datstat['time'] = datstat['time'].dt.time
        datstat.set_index('time',inplace=True)

        # data from previous day
        datprev = dat[time.between(ts,te)]
        datprev = datprev.set_axis(datprev.index.time)
        
    xlm = [dt.time(0,0),dt.time(23,59)]

    clrindx = np.round(np.linspace(0,255,len(dat)))

    if p is None:
        p = figure(title=title,x_axis_type="datetime", x_axis_label=xlb, y_axis_label=ylb,
        x_range=xlm,y_range=ylm,aspect_ratio =1,width = 300,tools=TOOLS, toolbar_location='below')
        p.add_tools(HoverTool(
        tooltips=[
            ('Time', '@x{%T}'),
            ('Value','@y')
        ],
        formatters={
            '@x': 'datetime',
        }
        ))  
        # removing grid lines
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.outline_line_color = "black"
        p.outline_line_width = 1
        # adding zero line
        p.line(xlm, [0,0], line_width=1,line_color='grey',line_dash='dashed')

    if len(dat)>0:
        # median (25th/75th percentile) diel pattern during the past days
        source = ColumnDataSource(dict(x=datstat.index, y1=datstat['25perc'], y2=datstat['75perc']))
        vararea = VArea(x="x", y1="y1", y2="y2", fill_color="grey",fill_alpha=0.5)
        p.add_glyph(source,vararea)
        p.line(time2.time, datstat['median'], line_width=1,line_color='grey')

        # data from the past day
        p.line(datprev.index, datprev, line_width=1.4,line_color='red')
    
    # show the results
    if saveshowflag==1:
        # save
        if len(flout)>0:
            save(p,flout)
    elif saveshowflag==2:
        # show
        show(p)
    # elif saveshowflag==0:
    #     # only return
    
    return p

def PlotBoxPlot(group,y,lglbl,lclr='black',fclr='grey',p=None,title='',xlm=None,ylm=None,xlb='',ylb=''):

    cats = group.unique()
    df = pd.DataFrame(dict(y=y, group=group))



    if p is None:
        
        # no figure as input
        TOOLS = "save,pan,box_zoom,reset,wheel_zoom"
        p = figure(title=title, x_axis_label=xlb, y_axis_label=ylb,
        x_range=xlm,y_range=ylm,aspect_ratio =2.5,width = 800,tools=TOOLS, toolbar_location='above')


    # find the quartiles and IQR for each category
    groups = df.groupby('group')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.y > upper.loc[cat]['y']) | (group.y < lower.loc[cat]['y'])]['y']
    out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = list(out.index.get_level_values(0))
        outy = list(out.values)


    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.y = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'y']),upper.y)]
    lower.y = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'y']),lower.y)]

    # stems
    p.segment(cats, upper.y, cats, q3.y, line_color=lclr)
    p.segment(cats, lower.y, cats, q1.y, line_color=lclr)

    # boxes
    p.vbar(cats, np.median(np.diff(cats))*0.8, q2.y, q3.y, fill_color=fclr, line_color=lclr, legend_label=lglbl,fill_alpha = 0.5)
    p.vbar(cats, np.median(np.diff(cats))*0.8, q1.y, q2.y, fill_color=fclr, line_color=lclr,fill_alpha = 0.5)


    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower.y, np.median(np.diff(cats))*0.8, 0.000001, line_color=lclr, fill_color=lclr)
    p.rect(cats, upper.y, np.median(np.diff(cats))*0.8, 0.000001, line_color=lclr, fill_color=lclr)

    # median
    p.circle(cats, q2.y, size=6, color=lclr, fill_alpha=0.6)
    # p.rect(cats, q2.y, np.median(np.diff(cats))*0.8, 0.000001, line_color='red', fill_color='red')

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color=lclr, fill_alpha=0.6)


    return p

def PlotScatter2(x,y,time,title='',ylb='',xlb='',flout='',saveshowflag=2,ylm=[],xlm=[],plot11=False,plotfit=False,lclr='black',fclr='grey'):

    # defining x-axis limits
    if len(xlm)<2:

        MAD = np.nanmedian(np.abs(x-np.nanmedian(x)))
        xb = np.nanpercentile(x,1)-MAD*1
        xt = np.nanpercentile(x,99)+MAD*1
        xlm = [xb,xt]

    # defining y-axis limits using the data between xlm
    if len(ylm)<2:

        MAD = np.nanmedian(np.abs(y-np.nanmedian(y)))
        yb = np.nanpercentile(y,1)-MAD*1
        yt = np.nanpercentile(y,99)+MAD*1
        ylm = [yb,yt]
            
    data = pd.DataFrame({'x':x,'y':y,'time':time})
    # scatter = hv.Scatter(data, vdims=['x', 'y', 'time'])
    # scatter = hv.Scatter(data)
    scatter = hv.Scatter((x,y,time),kdims='x',vdims=['y','z'],label='Data')
    # scatter = scatter.opts(color='z',fill_alpha=0.8,line_color='black')
    scatter = scatter.opts(fill_alpha=0.8,line_color='black')
    # curve = hv.Curve(data, vdims=['x', 'y'])
    curve = hv.Curve((x,y),kdims='x',vdims=['y'])
    # data11 = pd.DataFrame({'x':xlm,'y':ylm})
    fig = scatter

    if plot11:
        # curve11 = hv.Curve(data11, vdims=['x', 'y'])
        # curve11 = hv.Curve(data11)
        curve11 = hv.Curve((xlm,xlm),label='1:1 line')
        curve11 = curve11.opts(line_color='grey',line_dash='dashed')
        fig = fig*curve11


    if plotfit:
        
        def linear_func(B, x):
            return B[0]*x+B[1]
        linear = odr.Model(linear_func)
        dat = odr.Data(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
        odr1 = odr.ODR(dat, linear, beta0=[1.0, 0.0])
        out = odr1.run()
        # out.pprint()
        slope = out.beta[0]
        intercept = out.beta[1]

        # res = stats.linregress(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
        # slope = res[0]
        # intercept = res[1]
        
        datafit = pd.DataFrame({'x':np.array(xlm),'y':np.array(xlm)*slope + intercept})
        # curvefit = hv.Curve(datafit, vdims=['x', 'y'])
        # curvefit = hv.Curve(datafit)
        curvefit = hv.Curve((np.array(xlm),np.array(xlm)*slope + intercept),label='Fit:y=ax+b\n\n'+'a='+f"{slope:.2f}"+'\n\n'+'b='+f"{intercept:.2f}")
        curvefit = curvefit.opts(line_color='red')
        fig = fig*curvefit

        # regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        # p.add_layout(regression_line)

        # p.line(xlm,np.array(xlm)*slope + intercept, line_width=1,line_color='red', legend_label='Fit:y=ax+b\n\n'+'a='+f"{slope:.2f}"+'\n\n'+'b='+f"{intercept:.2f}")

    fig.redim.label(x=xlb, y=ylb)
    
    return fig
    

def PlotSpectra(freq,Px,Py=None,rmfrqweight=False,title='',ylb='fP_xx/var',xlb='Frequency (Hz)',flout='',xvarlbl='',yvarlbl='',saveshowflag=2,ylm=[],xlm=[],plotnoise=False,rmnoise=False,lclr='black',fclr='grey'):

    if rmfrqweight:
        Px = Px/freq

    # defining x-axis limits
    if len(xlm)<2:
        xlm = [np.min(freq),np.max(freq)]

    # defining y-axis limits using the data between xlm
    if len(ylm)<2:

        MAD = np.nanmedian(np.abs(Px-np.nanmedian(Px)))
        yb = np.nanpercentile(Px,1)-MAD*1
        yt = np.nanpercentile(Px,99)+MAD*1
        ylm = [yb,yt]
            

    data = pd.DataFrame({'x':freq,'y':Px})
    # scatter = hv.Scatter(data, vdims=['x', 'y', 'time'])
    # scatter = hv.Scatter(data)    
    curvex = hv.Curve((freq,Px),kdims='freq',vdims='Px',label=xvarlbl)
    fig = curvex
    if Py is not None:
        if rmfrqweight:
            Py = Py/freq
        curvey = hv.Curve((freq,Py),kdims='freq',vdims='Py',label=yvarlbl)
        fig = fig*curvey




    if rmnoise:
        noise = Px[freq==np.max(freq)]
        noisevec = np.ones(np.shape(Px))*noise
        if not rmfrqweight:
            noisevec = noisevec*freq/np.max(freq)
        curvexn = hv.Curve((freq,Px-noisevec),kdims='freq',vdims='Px',label=xvarlbl)
        fig = fig*curvexn



    if plotnoise:
        noise = Px[freq==np.max(freq)]
        noise = np.append(noise,noise)
        if not rmfrqweight:
            noise = noise*np.append(np.min(freq)/np.max(freq),1)
        # curve11 = hv.Curve(data11, vdims=['x', 'y'])
        # curve11 = hv.Curve(data11)
        curvenoise = hv.Curve((np.append(np.min(freq),np.max(freq)),noise),label='white noise')
        curvenoise = curvenoise.opts(line_color='grey',line_dash='dashed')
        fig = fig*curvenoise


    # if plotfit:
        
    #     def linear_func(B, x):
    #         return B[0]*x+B[1]
    #     linear = odr.Model(linear_func)
    #     dat = odr.Data(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
    #     odr1 = odr.ODR(dat, linear, beta0=[1.0, 0.0])
    #     out = odr1.run()
    #     # out.pprint()
    #     slope = out.beta[0]
    #     intercept = out.beta[1]

    #     # res = stats.linregress(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
    #     # slope = res[0]
    #     # intercept = res[1]
        
    #     datafit = pd.DataFrame({'x':np.array(xlm),'y':np.array(xlm)*slope + intercept})
    #     # curvefit = hv.Curve(datafit, vdims=['x', 'y'])
    #     # curvefit = hv.Curve(datafit)
    #     curvefit = hv.Curve((np.array(xlm),np.array(xlm)*slope + intercept),label='Fit:y=ax+b\n\n'+'a='+f"{slope:.2f}"+'\n\n'+'b='+f"{intercept:.2f}")
    #     curvefit = curvefit.opts(line_color='red')
    #     fig = fig*curvefit

    #     # regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
    #     # p.add_layout(regression_line)

    #     # p.line(xlm,np.array(xlm)*slope + intercept, line_width=1,line_color='red', legend_label='Fit:y=ax+b\n\n'+'a='+f"{slope:.2f}"+'\n\n'+'b='+f"{intercept:.2f}")

    # fig.redim.label(x=xlb, y=ylb)
    fig.opts(xlabel=xlb,ylabel=ylb,title=title,logx=True, logy=True)
    if Py is not None:
        fig.opts(show_legend=True,legend_position='bottom_left')
    # fig.redim(y=hv.Dimension('y', range=ylm))
    return fig
    

def PlotTransferFunction(freq,Px,Py,title='',ylb='TRF',xlb='Frequency (Hz)',powcoflag=True,ylm=[],xlm=[],plotfit=True,plotspec=False,normalise=False,fnormlim=[1e-2,1e-1],ffitlim=[1e-2,3]):


    # defining x-axis limits
    if len(xlm)<2:
        xlm = (np.min(freq),np.max(freq))

    if normalise:

        nxvar = np.trapz(Px[(freq>fnormlim[0]) & (freq<fnormlim[1])], freq[(freq>fnormlim[0]) & (freq<fnormlim[1])])
        Px = Px/nxvar

        nyvar = np.trapz(Py[(freq>fnormlim[0]) & (freq<fnormlim[1])], freq[(freq>fnormlim[0]) & (freq<fnormlim[1])])
        Py = Py/nyvar

    TRF = Px/Py
    # defining y-axis limits using the data between xlm
    if len(ylm)<2:
        ylm = (-0.1,1.2)
            
    curvex = hv.Scatter((freq,TRF),kdims='freq',vdims='TRF',label='Data')
    curvex.opts(fill_alpha=0.8,line_color='black',width=500, height=500)
    fig = curvex




    if plotfit:
        x = freq
        y = TRF
        def sigmoidal(B, x):
            return B[0]/(1+(2*np.pi*x*B[1])**2)

        def sigmoidalsqrt(B, x):
            return B[0]/(1+(2*np.pi*x*B[1])**2)**0.5

        def sigmoidalphase(B, x):
            return B[0]*(np.cos(-2*np.pi*x*B[2])-2*np.pi*x*B[1]*np.sin(-2*np.pi*x*B[2]))/(1+(2*np.pi*x*B[1])**2)

        
        if powcoflag:
            linear = odr.Model(sigmoidal)        

            indx = (~np.isnan(x*y)) & ((x>ffitlim[0]) & (x<ffitlim[1])) & (TRF>-0.5) & (TRF<1.5)
            dat = odr.Data(x[indx],y[indx])
            odr1 = odr.ODR(dat, linear, beta0=[1,0.1])
            out = odr1.run()
            tau = out.beta[1]

            TRFfit = 1/(1+(2*np.pi*freq*tau)**2)
            curvefit = hv.Curve((freq,TRFfit),kdims='freq',vdims='TRF',label='H, tau='+f"{tau:.2f}"+'s')        

            fig = fig*curvefit

        else:
            linear = odr.Model(sigmoidalsqrt) 
            linear = odr.Model(sigmoidal)        
            linear2 = odr.Model(sigmoidalphase)

            indx = (~np.isnan(x*y)) & ((x>ffitlim[0]) & (x<ffitlim[1])) & (TRF>-0.5) & (TRF<1.5)
            dat = odr.Data(x[indx],y[indx])
            odr1 = odr.ODR(dat, linear, beta0=[1,0.1])
            odr2 = odr.ODR(dat, linear2, beta0=[1,0.1,0])
            out = odr1.run()
            out2 = odr2.run()
            tau = out.beta[1]

            TRFfit = 1/(1+(2*np.pi*freq*tau)**2)**0.5
            TRFfit = 1/(1+(2*np.pi*freq*tau)**2)
            TRFfit2 = (np.cos(-2*np.pi*freq*out2.beta[2])-2*np.pi*freq*out2.beta[1]*np.sin(-2*np.pi*freq*out2.beta[2]))/(1+(2*np.pi*freq*out2.beta[1])**2)
            curvefit = hv.Curve((freq,TRFfit),kdims='freq',vdims='TRF',label='H**0.5, tau='+f"{tau:.2f}"+'s')        
            curvefit2 = hv.Curve((freq,TRFfit2),kdims='freq',vdims='TRF2',label='HH_p, tau='+f"{out2.beta[1]:.2f}"+'s'+', tlpf='+f"{out2.beta[2]:.2f}"+'s')

            fig = fig*curvefit*curvefit2

    if plotspec:
        curvex = hv.Curve((freq,Px),kdims='freq',vdims='Px',label='Gas')
        curvey = hv.Curve((freq,Py),kdims='freq',vdims='Py',label='T')
        

        if powcoflag:
            curvey2 = hv.Curve((freq,Py*TRFfit),kdims='freq',vdims='Py*TRFfit',label='T*H')
            ylbl2 = 'fP_xx/var'
            fig2 = curvex*curvey*curvey2
        else:
            curvey2 = hv.Curve((freq,Py*TRFfit),kdims='freq',vdims='Py*TRFfit',label='T*H**0.5')
            curvey3 = hv.Curve((freq,Py*TRFfit2),kdims='freq',vdims='Py*TRFfit2',label='T*HH_p')
            ylbl2 = 'fCo/cov'
            fig2 = curvex*curvey*curvey2*curvey3

        fig2.opts(xlabel=xlb,ylabel=ylbl2,title=title,logx=True,logy=False,xlim=xlm,width=500, height=500)
        fig2.opts(show_legend=True,legend_position='bottom_left')
    # if plotfit:
        
    #     def linear_func(B, x):
    #         return B[0]*x+B[1]
    #     linear = odr.Model(linear_func)
    #     dat = odr.Data(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
    #     odr1 = odr.ODR(dat, linear, beta0=[1.0, 0.0])
    #     out = odr1.run()
    #     # out.pprint()
    #     slope = out.beta[0]
    #     intercept = out.beta[1]

    #     # res = stats.linregress(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
    #     # slope = res[0]
    #     # intercept = res[1]
        
    #     datafit = pd.DataFrame({'x':np.array(xlm),'y':np.array(xlm)*slope + intercept})
    #     # curvefit = hv.Curve(datafit, vdims=['x', 'y'])
    #     # curvefit = hv.Curve(datafit)
    #     curvefit = hv.Curve((np.array(xlm),np.array(xlm)*slope + intercept),label='Fit:y=ax+b\n\n'+'a='+f"{slope:.2f}"+'\n\n'+'b='+f"{intercept:.2f}")
    #     curvefit = curvefit.opts(line_color='red')
    #     fig = fig*curvefit

    #     # regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
    #     # p.add_layout(regression_line)

    #     # p.line(xlm,np.array(xlm)*slope + intercept, line_width=1,line_color='red', legend_label='Fit:y=ax+b\n\n'+'a='+f"{slope:.2f}"+'\n\n'+'b='+f"{intercept:.2f}")

    # fig.redim.label(x=xlb, y=ylb)
    fig.opts(xlabel=xlb,ylabel=ylb,title=title,logx=True)
    fig.opts(show_legend=True,legend_position='bottom_left')
    fig.opts(xlim=xlm,ylim=ylm)
    # fig.redim(y=hv.Dimension('y', range=ylm))

    if plotspec:
        fig = (fig + fig2).opts(shared_axes=False)

    return fig
    



def PlotScatter(x,y,time,title='',ylb='',xlb='',flout='',saveshowflag=2,ylm=[],xlm=[],zlm=[],plot11=False,plotzero=False,plotfit=False,lclr='black',fclr='grey'):

    TOOLS = "save,pan,box_zoom,reset,wheel_zoom"

    # defining x-axis limits
    if len(xlm)<2:

        MAD = np.nanmedian(np.abs(x-np.nanmedian(x)))
        xb = np.nanpercentile(x,1)-MAD*1
        xt = np.nanpercentile(x,99)+MAD*1
        xlm = [xb,xt]

    # defining y-axis limits using the data between xlm
    if len(ylm)<2:

        MAD = np.nanmedian(np.abs(y-np.nanmedian(y)))
        yb = np.nanpercentile(y,1)-MAD*1
        yt = np.nanpercentile(y,99)+MAD*1
        ylm = [yb,yt]
            

    if len(zlm)<2 and time is not None:
        zlm = [np.min(time),np.max(time)]
            


    p = figure(title=title, x_axis_label=xlb, y_axis_label=ylb,
    x_range=xlm,y_range=ylm,aspect_ratio =2,width = 1200,tools=TOOLS, toolbar_location='above')
    p.add_tools(HoverTool(
    tooltips=[('x', '@x'),('y','@y')]))  
    # removing grid lines
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.outline_line_color = "black"
    p.outline_line_width = 1



    
    if time is not None:
        source = ColumnDataSource(data={
            'x'       : x,
            'y'       : y,
            'time'      : time.astype('int64') // 10**9,
        })
        # mapper = linear_cmap(field_name='time', palette=cm.batlow, low=zlm[0].timestamp(), high=zlm[1].timestamp())
        mapper = linear_cmap(field_name='time', palette=Viridis256, low=zlm[0].timestamp(), high=zlm[1].timestamp())
        p.scatter(x='x', y='y', marker='circle', fill_color=mapper, line_color='black', source=source,fill_alpha=0.6, size=10)

        # color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0), ticker=BasicTicker(), formatter = DatetimeTickFormatter(days='%d/%m'), label_standoff=12)
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0),
         ticker=BasicTicker(), formatter = DatetimeTickFormatter(days='%d/%m',minutes = '%d/%m'))
        # color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), formatter = DatetimeTickFormatter(), label_standoff=12, border_line_color=None, location=(0,0))
        p.add_layout(color_bar, 'right')
    else:
        source = ColumnDataSource(data={
            'x'       : x,
            'y'       : y})
        p.scatter(x='x', y='y', marker='circle', fill_color='grey', line_color='black', source=source,fill_alpha=0.6, size=10)


    if plot11:
        p.line(xlm,xlm, line_width=1,line_color='grey',line_dash='dashed')
    if plotzero:
        p.line(xlm,[0,0], line_width=1,line_color='grey',line_dash='dashed')

    if plotfit:
        
        def linear_func(B, x):
            return B[0]*x+B[1]
        linear = odr.Model(linear_func)
        dat = odr.Data(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
        odr1 = odr.ODR(dat, linear, beta0=[1.0, 0.0])
        out = odr1.run()
        # out.pprint()
        slope = out.beta[0]
        intercept = out.beta[1]

        # res = stats.linregress(x[~np.isnan(x*y)],y[~np.isnan(x*y)])
        # slope = res[0]
        # intercept = res[1]
        
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        p.add_layout(regression_line)

        # p.line(xlm,np.array(xlm)*slope + intercept, line_width=1,line_color='red', legend_label='Fit:y=ax+b\n\n'+'a='+f"{slope:.2f}"+'\n\n'+'b='+f"{intercept:.2f}")

    # def update_plot(attr,old,new):
    #     dates = slider.value_as_datetime

    #     source.data = new_data

    # slider = DateRangeSlider(value=(np.min(time), np.max(time)), start=np.min(time), end=np.max(time), format="%x, %X")
    # slider.on_change('value', update_plot)

    p.legend.location = "top_left"

    # l = column(p,slider)
    # l = column(p)
    
    # show the results
    if saveshowflag==1:
        # save
        if len(flout)>0:
            save(p,flout)
    elif saveshowflag==2:
        # show
        show(p)
    # elif saveshowflag==0:
    #     # only return
    
    return p


def PlotFpr(fpr,lat,lon,xlm,ylm,flout='',saveshowflag=2):

    TRAN_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    x,y = TRAN_4326_TO_3857.transform(lat, lon)


    # cntrs = gpd.GeoDataFrame(crs="EPSG:3857")
    cntrs2 = list()
    for indx in range(len(fpr['xr'])):
        xp = [x1+x for x1 in fpr['xr'][indx]]
        yp = [y1+y for y1 in fpr['yr'][indx]]
        # cntrs = cntrs.append({'geometry':sgeom.LineString(zip(xp, yp)),'percentile':fpr['rs'][indx]}, ignore_index = True)

        cntrs2.append({'x':xp,'y':yp,'percentile':fpr['rs'][indx]})

    # tiles = gvts.EsriImagery
    # # lcontours = gv.LineContours((fpr['x_2d']+x, fpr['y_2d']+y, csf),crs=ccrs.GOOGLE_MERCATOR).opts(line_color='black',line_width=2)
    # # fcontours = gv.FilledContours((fpr['x_2d']+x, fpr['y_2d']+y, csf),crs=ccrs.GOOGLE_MERCATOR).opts(fill_alpha=0.3,line_color=None,cmap='PuBu')
    # fprplt = gv.Polygons(cntrs, crs=ccrs.GOOGLE_MERCATOR, vdims='percentile').opts(fill_alpha=0.2,line_color='black',tools=['hover'])

    # site = gv.Points((x,y),crs=ccrs.GOOGLE_MERCATOR).opts(size=5,line_color="black",fill_color='white')
    # p = (tiles*fprplt*site).opts(width=500, height=500, xaxis=None, yaxis=None,xlim=(xlm[0]+x,xlm[1]+x), ylim=(ylm[0]+y,ylm[1]+y), aspect='equal')

    # pfig = pn.Column(p)
    # pfig.save('D:/fpr.html', resources=INLINE)


    tiles = hv.element.tiles.EsriImagery()
    # fprplt = hv.Polygons(cntrs, vdims='percentile').opts(fill_alpha=0.2,line_color='black',tools=['hover'])
    fprplt = hv.Polygons(cntrs2, vdims='percentile').opts(fill_alpha=0.2,line_color='black',tools=['hover'])
    site = hv.Points((x,y)).opts(size=5,line_color="black",fill_color='white')
    p = (tiles*fprplt*site).opts(width=500, height=500,show_grid=False, xaxis=None, yaxis=None,xlim=(xlm[0]+x,xlm[1]+x), ylim=(ylm[0]+y,ylm[1]+y), aspect='equal')

    pfig = pn.Column(p)


    # show the results
    if saveshowflag==1:
        # save
        if len(flout)>0:
            save(pfig,flout)
            pfig.save('D:/fpr2.html')
    elif saveshowflag==2:
        pfig.show()
    # elif saveshowflag==0:
    #     # only return
    
    return p