#%%
import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import pyplot as plt
import churn_analysis as utils
import os

label = 'Underdog Win'
cols_drop = ['unique_game_id', 'DATE', 'Home Team', 'Away Team']

main_path = '..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows\\'
filename = '\\2015_stats_n10.csv'

def construct_df(years, n):
    '''
    Given that our files are located in "..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows",
    return concatenated df for a given rolling avg window (n) and a list of years
    
    years: type list of ints from 2009 - 2020
    n: type in from 5 to 50 inclusive
    '''
    ## Maybe special case for dfs including 2013- and 2014+ need to be included
    filenames = [str(year) + "_stats_n" + str(n) + ".csv" for year in years]
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(main_path + filename))

    concatenated_df = pd.concat(dfs).drop(columns=cols_drop)
    return concatenated_df


def get_top_mutual_info(df, num_top):
    X = df_2015.drop(label, 1)
    Y = df_2015[[label]].values
    cols = X.columns.values
    mis = []

    # Get MI
    for c in cols:
        mis.append(sk.metrics.normalized_mutual_info_score(Y.ravel(), X[[c]].values.ravel()))

    top_mis = sorted(list(zip(mis, cols)), key = lambda x: x[0])[-num_top:]
    bottom_mis = sorted(list(zip(mis, cols)), key = lambda x: x[0])[:len(mis) - num_top]
    return top_mis, bottom_mis

# %%
## we should check which are the top features for most n vals and many seasons...