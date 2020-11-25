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
    X = df.drop(label, 1)
    Y = df[[label]].values
    cols = X.columns.values
    mis = []

    # Get MI
    for c in cols:
        mis.append(sk.metrics.normalized_mutual_info_score(Y.ravel(), X[[c]].values.ravel()))

    top_mis = sorted(list(zip(mis, cols)), key = lambda x: x[0])[-num_top:]
    bottom_mis = sorted(list(zip(mis, cols)), key = lambda x: x[0])[:len(mis) - num_top]
    return top_mis, bottom_mis

def num_times_in_top(years, num_top):
    '''
    For all n in range 5 to 50, count the number of times a feature is in the top
    num_top features with respect to mutual information score. Assuming for now that 
    years span 2014-2019, OR 2009-2013 (due to differences in dfs)
    '''
    ## Making a dummy df for simplicity
    df = construct_df(years, 5)
    freq_dict = {c:0 for c in df.columns.values}
    for n in range(5, 51):
        df = construct_df(years, n)
        top, bottom = get_top_mutual_info(df, num_top)
        for mi, c in top:
            freq_dict[c] += 1

    return freq_dict

# %%
## Below are the frequencies of the features being in the top num_top "most informative"
## though I'm sure all the MIs are close (there seemed to be a hard drop at around 0.1 MI)
years = list(range(2014, 2020))
freq = num_times_in_top(years, num_top=20)
freq_list = sorted(list(freq.items()), key=lambda x: x[1])


# %%
## Some plots if we want
'''
#%%
ind = 30
fig = plt.figure(figsize = (20, 15))
ax = plt.subplot(111)
plt.subplots_adjust(bottom = 0.25)
#ax.set_xticks(ind + getTickAdj([x[1] for x in topmis], 1))
ax.set_xticklabels([x[1] for x in topmis], rotation = 45, size = 14)
plt.bar([x[1] for x in topmis], [x[0] for x in topmis])
'''
