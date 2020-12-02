#%%
import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import pyplot as plt
import churn_analysis as utils
import os

label = 'Underdog Win'
label_ind = 10 ## After you drop the columns
std_cols_drop = ['unique_game_id', 'DATE', 'Home Team', 'Away Team', 
                 'Away All Stars', 'Home All Stars', 'HOME_IP_AVAIL',
                 'AWAY_IP_AVAIL']

additional_drop = ['Home Team Cut_offensive',
       'Away Team Cut_offensive', 'Home Team Handoff_defensive',
       'Away Team Handoff_defensive', 'Home Team Handoff_offensive',
       'Away Team Handoff_offensive', 'Home Team Isolation_defensive',
       'Away Team Isolation_defensive', 'Home Team Isolation_offensive',
       'Away Team Isolation_offensive', 'Home Team Misc_offensive',
       'Away Team Misc_offensive', 'Home Team OffScreen_defensive',
       'Away Team OffScreen_defensive', 'Home Team OffScreen_offensive',
       'Away Team OffScreen_offensive', 'Home Team PnRBH_defensive',
       'Away Team PnRBH_defensive', 'Home Team PnRBH_offensive',
       'Away Team PnRBH_offensive', 'Home Team PnRRM_defensive',
       'Away Team PnRRM_defensive', 'Home Team PnRRM_offensive',
       'Away Team PnRRM_offensive', 'Home Team Postup_defensive',
       'Away Team Postup_defensive', 'Home Team Postup_offensive',
       'Away Team Postup_offensive', 'Home Team Putbacks_defensive',
       'Away Team Putbacks_defensive', 'Home Team Putbacks_offensive',
       'Away Team Putbacks_offensive', 'Home Team Spotup_defensive',
       'Away Team Spotup_defensive', 'Home Team Spotup_offensive',
       'Away Team Spotup_offensive', 'Home Team Transition_defensive',
       'Away Team Transition_defensive', 'Home Team Transition_offensive',
       'Away Team Transition_offensive']

non_relative_columns = ['AWAY_CUM_WIN_PCT', 'AWAY_PRIOR_WIN_V_OPP',
                        'AWAY_ROLL_WIN_PCT', 'Away 3P', 'Away 3P%', 'Away 3PAr', 'Away AST',
                        'Away AST%', 'Away BLK', 'Away BLK%', 'Away DRB',
                        'Away DRB%', 'Away DRtg', 'Away FG', 'Away FG%', 'Away FT', 'Away FT%',
                        'Away FTr', 'Away ORB', 'Away ORB%', 'Away ORtg', 'Away Odds Close',
                        'Away PF', 'Away STL', 'Away STL%', 'Away TOV', 'Away TOV%', 'Away TS%',
                        'Away eFG%', 'HOME_CUM_WIN_PCT', 'HOME_PRIOR_WIN_V_OPP',
                        'HOME_ROLL_WIN_PCT', 'Home 3P', 'Home 3P%', 'Home 3PAr', 'Home AST',
                        'Home AST%', 'Home BLK', 'Home BLK%', 'Home DRB',
                        'Home DRB%', 'Home DRtg', 'Home FG', 'Home FG%', 'Home FT', 'Home FT%',
                        'Home FTr', 'Home ORB', 'Home ORB%', 'Home ORtg', 'Home Odds Close',
                        'Home PF', 'Home STL', 'Home STL%', 'Home TOV', 'Home TOV%', 'Home TS%',
                        'Home eFG%']

#main_path = '..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows\\'
#filename = '\\2015_stats_n10.csv'

def construct_df(years, n, main_path, drop_playstyle=True):
    '''
    Given that our files are located in "..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows",
    return concatenated df for a given rolling avg window (n) and a list of years
    
    years: type list of ints from 2009 - 2020
    n: type in from 5 to 50 inclusive
    '''
    
    filenames = [str(year) + "_stats_n" + str(n) + ".csv" for year in years]
    dfs = []

    for filename in filenames:
        dfs.append(pd.read_csv(main_path + filename))
    if drop_playstyle:
        cols_drop = std_cols_drop + additional_drop
    ## Currently assuming that we are dropping playstyle stuff so wont create robust working for it
    ## remember to drop 'Away Underdog' after all of this is done

    concatenated_df = pd.concat(dfs).drop(columns=cols_drop)
    concatenated_df['Away Underdog'].replace(0, -1, inplace=True)

    new_column_tuples = []
    for i in range(int(len(non_relative_columns)/2)):
        new_column_tuples.append((non_relative_columns[i],non_relative_columns[i+29]))

    for col_pair in new_column_tuples:
        concatenated_df['underdog_rel_' + col_pair[0][5:]] = \
            (concatenated_df[col_pair[0]] - concatenated_df[col_pair[1]])*concatenated_df['Away Underdog']

    concatenated_df['Home Relative Age Diff'] = concatenated_df['Home Relative Age Diff']*concatenated_df['Away Underdog']*1 
    concatenated_df.drop(columns=non_relative_columns, inplace=True)

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
# years = list(range(2014, 2020))
# freq = num_times_in_top(years, num_top=20)
# freq_list = sorted(list(freq.items()), key=lambda x: x[1])


def plot_correlation_matrix(df, features_drop_list):
    '''
    Plot correlation matrix given a dataframe. Give list of features you want to 
    exclude or not. If include_label is True, then the label will not be dropped in the
    correlation matrix.

    Preferred input style of features_drop_list: df.columns.values[indices] so as to
    not list out the features every single time
    '''
    ## df.columns.values[65:] ## cuts out all team playstyle stuff

    features_drop_list = list(features_drop_list) ## Some error prevention

    # if not include_label:
    #     features_drop_list.append(label)
    if features_drop_list:
        df = df.drop(columns=features_drop_list)

    utils.plotCorr(df, label, 20, 20)

    return None





'''
# %%
## Some plots if we want

#%%
ind = 30
fig = plt.figure(figsize = (20, 15))
ax = plt.subplot(111)
plt.subplots_adjust(bottom = 0.25)
#ax.set_xticks(ind + getTickAdj([x[1] for x in topmis], 1))
ax.set_xticklabels([x[1] for x in topmis], rotation = 45, size = 14)
plt.bar([x[1] for x in topmis], [x[0] for x in topmis])
'''
