# Import all the needed libraries
from IPython.core.display import display, HTML
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from time import time
import shap
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from IPython.display import display, HTML
import csv
from itertools import islice
from PIL import Image
from functools import reduce
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy.polynomial.polynomial as poly
from dateutil import relativedelta
import math
import warnings
import matplotlib.ticker as plticker
from matplotlib.ticker import MaxNLocator
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.stats import norm, skew, kurtosis
from scipy import stats
from calendar import isleap
import datetime
import time
import os
import pandas as pd
import numpy as np
#from skimpy import skim
import pprint
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error






# to center figures and tables throughout the report
display(HTML("""<style>.output {
    display: flex;
    align-items: center;
    text-align: center;}</style> """))


####
shap.initjs()

# 
def status(df):

	table = [[i,
           len(df[i]), df[i].isna().sum(),
            "{:.1%}".format(df[i].isna().sum()/len(df[i]))]
          for i in df.columns]
	headers = ['Features', 'Observations', 'No of missing', '% Missing ']
	print(tabulate(table, headers, tablefmt='pretty', numalign='center'))

#
def sequence_of_missing_values(df,feature):
    """Create the table of the missing range"""
    df = df.replace(np.nan, -996)
    table = [[v.index[0],v.index[-1],len(v)]
    for k, v in df[df[feature] == -996].groupby((df[feature] != -996).cumsum())]


    df_missing = pd.DataFrame(table, columns=['Start_Date',
                                             'End_Date','Frequency'])

    df_missing['Start_Date'] = df_missing['Start_Date'].dt.strftime(
    	"'%Y-%m-%d'")
   
    df_missing['End_Date'] = df_missing['End_Date'].dt.strftime(
    	"'%Y-%m-%d'")

    return df_missing.sort_values(by=['Frequency'], ascending=False).head(20)

#
def corr_plot(df, Title):
    f, ax = plt.subplots(figsize=(10, 8))

    corr = df.corr()
    mask = np.zeros_like(corr, dtype=None)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, cmap='coolwarm', mask=mask, annot=True, )

    plt.title(Title)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12, rotation=0)
    plt.show()

#

def display_side_by_side(dfs: list, captions: list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes(
            "style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))

    
def multiplot_SM_vs_ClimateVar(m,n, time_start, time_end, df, var_list, clim_var_index, clim_var_ylabel, ylim_min, \
                               ylim_max, ylim_twin_min, ylim_twin_max, leg_loc, xtic_rot, font_size):
    df_SM_data  = []
    df_SM_data  = df[var_list]
    
    # ---- plot     
    i = 0    
    while i < m * n:
        i=i+1
        df_SM_data_seg         = df_SM_data.loc[time_start[i-1] : time_end[i-1], var_list]
        df_SM_data_seg['Date'] = pd.to_datetime(df_SM_data_seg.index)
        ax=fig.add_subplot(m,n,i)
        ax.plot(df_SM_data_seg['Date'], df_SM_data_seg[var_list[4]], color = 'orangered', label = '5 cm') 
        ax.plot(df_SM_data_seg['Date'], df_SM_data_seg[var_list[5]], color = 'forestgreen', label = '25 cm') 
        ax.plot(df_SM_data_seg['Date'], df_SM_data_seg[var_list[6]], color = 'black', label = '45 cm') 
        ax.tick_params(axis = 'x', which = 'major', rotation = xtic_rot[i-1], labelsize = font_size-2)
        ax.tick_params(axis = 'y', which = 'major', labelsize = font_size-2)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_xlabel("Year", fontsize = font_size)
        ax.set_ylabel("Soil moisture (%)",  fontsize = font_size)
        ax.xaxis.set_tick_params(pad=5)
        ax.set_ylim([ylim_min[i-1], ylim_max[i-1]])
        leg = ax.legend(loc = leg_loc[i-1],prop={'size': font_size-6})
    
        # -- second/twin axis
        ax2 = ax.twinx()
        ax2.plot(df_SM_data_seg['Date'], df_SM_data_seg[var_list[clim_var_index]], color = 'dodgerblue', label = 'precip') 
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.set_ylabel(clim_var_ylabel, fontsize = font_size)
        ax2.set_ylim([ylim_twin_min[i-1], ylim_twin_max[i-1]])
        ax2.tick_params(axis = 'y', which = 'major', labelsize = font_size-2)
        
    # --adjust horizontal and vertical spacing between the plots
        fig.subplots_adjust(wspace=0.6)
        fig.subplots_adjust(hspace=0.5)
     
    return df_SM_data_seg        
    #


def print_results_(pipeline, y_test):

    fig, ax = plt.subplots(1, 3, figsize=(14, 6))

    for count, model_name in enumerate(holdem):

        ax[count].scatter((y_test), (holdem[model_name]),
                          marker='o', color='black')

        RSQ = np.round((r2_score((y_test), (holdem[model_name]))), 3)

        ax[count].text(0.95, 0.1, ("$R^2$: %0.03f" % RSQ),
                       verticalalignment='bottom', horizontalalignment='right',
                       transform=ax[count].transAxes,
                       color='black', fontsize=21)

        ax[count].set_xlabel('Recorded', fontsize=18)
        ax[count].tick_params(axis="x", labelsize=14, rotation=34)
        ax[count].xaxis.set_tick_params(pad=5)
        ax[count].set_ylabel('Predicted', fontsize=18)
        ax[count].tick_params(axis="y", labelsize=14)
        ax[count].yaxis.set_tick_params(pad=5)
        ax[count].set_title(model_name, size=18)

        plt.tight_layout(pad=1.2)

    plt.show()




def print_results(pipeline, y_test):
    fig, ax = plt.subplots(1, 3, figsize=(14, 6))
# ###################################################################################################


    for count, model_name in enumerate(holdem):

        ax[count].scatter(np.sort(y_test), np.sort(holdem[model_name]),
                        marker='o', color='black')

        RSQ = np.round((r2_score(np.sort(y_test), np.sort(holdem[model_name]))), 3)

        ax[count].text(0.95, 0.1, ("$R^2$: %0.03f" % RSQ),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax[count].transAxes,
                    color='black', fontsize=21)

        ax[count].set_xlabel('Recorded', fontsize=18)
        ax[count].tick_params(axis="x", labelsize=14, rotation=34)
        ax[count].xaxis.set_tick_params(pad=5)
        ax[count].set_ylabel('Predicted', fontsize=18)
        ax[count].tick_params(axis="y", labelsize=14)
        ax[count].yaxis.set_tick_params(pad=5)
        ax[count].set_title(model_name, size=18)

        plt.tight_layout(pad=1.2)

plt.show()

def results(x, y, x_t, y_t, pipelines):
    table = PrettyTable()

    # Fit the pipelines
    [pipe.fit(x, y) for pipe in pipelines]

    pipe_dict = {0: 'RandomForest', 1: 'XGBoost', 2: 'ExtraTree'}

    results = [[pipe_dict[i],
                np.round(r2_score(y, model.predict(x)), decimals=2),
                np.round(np.sqrt(mean_squared_error(y, model.predict(x))), decimals=2)]
               for i, model in enumerate(pipelines)]
    table.title = 'Training set score'
    table.field_names = ['Algorithm', 'R-square', 'RMSE']
    table.add_rows(results)
    print(table)

    table = PrettyTable()

    #################################################################################################
    # Make predictions on testing data and test model accuracy
    #################################################################################################
    results2 = []
    model_results = {}
    i = 0
    count = 0
    for model in pipelines:
        history = np.empty(len(x_t))
        test_X = x_t[0].reshape(1, -1)
        history[0] = model.predict(test_X)

        for i in range(len(x_t)):

            #x_t[i+1][-2] = history[i]
            #x_t[i+1][-1] = x_t[i][-2]

            test_X = x_t[i+1].reshape(1, -1)
            history[i+1] = model.predict(test_X)
            i = i+1
            if i == (len(x_t)-1):
                break
        results2.append([model.steps[0][0],
                         np.round(
            r2_score((y_t),
                     (history)), decimals=2),
            np.round(
            np.sqrt(mean_squared_error((y_t),
                                       (history))), decimals=2)])
        model_results[model.steps[0][0]] = history
        count = count + 1

    table.title = 'Test set Score'
    table.field_names = ['Algorithm', 'R-square', 'RMSE']

    table.add_rows(results2)

    print(table)

    return model_results



    
