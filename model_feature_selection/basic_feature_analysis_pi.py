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


def get_top_mutual_info():

    return None

df_2015 = pd.read_csv(path).drop(columns=cols_drop)
#utils.plotCorr(df_2015, label, 18, 15)
utils.plotMI(df_2015, label, 0.05, 1)
# %%


X = df_2015.drop(label, 1)
Y = df_2015[[label]].values
cols = X.columns.values
mis = []

#Start by getting MI
for c in cols:
    mis.append(sk.metrics.normalized_mutual_info_score(Y.ravel(), X[[c]].values.ravel()))
# %%

for mi, col in zip(mis, cols):
    if mi > 0.1:
        print(np.around(mi, 3), col)
# %%
sorted(list(zip(mis, cols)), key = lambda x: x[0])
# %%
## we should check which are the top features for most n vals and many seasons...