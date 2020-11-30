# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:17:01 2020

@author: sanjs
"""

# Logistic Regression 

import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

cols_drop = ['unique_game_id', 'DATE', 'Home Team', 'Away Team']
non_relative_columns = ['AWAY_CUM_WIN_PCT', 'AWAY_IP_AVAIL','AWAY_PRIOR_WIN_V_OPP',
                        'AWAY_ROLL_WIN_PCT', 'Away 3P', 'Away 3P%', 'Away 3PAr', 'Away AST',
                        'Away AST%', 'Away All Stars', 'Away BLK', 'Away BLK%', 'Away DRB',
                        'Away DRB%', 'Away DRtg', 'Away FG', 'Away FG%', 'Away FT', 'Away FT%',
                        'Away FTr', 'Away ORB', 'Away ORB%', 'Away ORtg', 'Away Odds Close',
                        'Away PF', 'Away STL', 'Away STL%', 'Away TOV', 'Away TOV%', 'Away TS%',
                        'Away eFG%', 'HOME_CUM_WIN_PCT', 'HOME_IP_AVAIL', 'HOME_PRIOR_WIN_V_OPP',
                        'HOME_ROLL_WIN_PCT', 'Home 3P', 'Home 3P%', 'Home 3PAr', 'Home AST',
                        'Home AST%', 'Home All Stars', 'Home BLK', 'Home BLK%', 'Home DRB',
                        'Home DRB%', 'Home DRtg', 'Home FG', 'Home FG%', 'Home FT', 'Home FT%',
                        'Home FTr', 'Home ORB', 'Home ORB%', 'Home ORtg', 'Home Odds Close',
                        'Home PF', 'Home STL', 'Home STL%', 'Home TOV', 'Home TOV%', 'Home TS%',
                        'Home eFG%']

def construct_df(years, n, main_path):
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
    ## Currently assuming that we are dropping playstyle stuff so wont create robust working for it
    ## remember to drop 'Away Underdog' after all of this is done

    concatenated_df = pd.concat(dfs)#.drop(columns=cols_drop)
    concatenated_df.drop((cols_drop + list(concatenated_df.filter(regex = 'Home Team')) + list(concatenated_df.filter(regex = 'Away Team'))), axis = 1, inplace = True)
    concatenated_df['Away Underdog'].replace(0, -1, inplace=True)

    new_column_tuples = []
    for i in range(int(len(non_relative_columns)/2)):
        new_column_tuples.append((non_relative_columns[i],non_relative_columns[i+31]))

    for col_pair in new_column_tuples:
        concatenated_df['underdog_rel_' + col_pair[0][5:]] = \
            (concatenated_df[col_pair[0]] - concatenated_df[col_pair[1]])*concatenated_df['Away Underdog']

    concatenated_df['Home Relative Age Diff'] = concatenated_df['Home Relative Age Diff']*concatenated_df['Away Underdog']*1 
    concatenated_df.drop(columns=non_relative_columns, inplace=True)

    return concatenated_df


def portfolio_roi(truths, probs, stake, thresh=0.5):
    '''
    Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **stake - stake to bet on each game
    **thresh- confidence threshold for filtering bets
    returns an expected portfolio ROI
    '''
    
    # Get absolute odds from within function instead of passing them in as a parameter
    odds_df = pd.DataFrame()
    for year in [2019, 2020]:
        df = pd.read_csv(os.getcwd() + '\\' + f'{year}_stats_n5.csv')
        odds_df = pd.concat([odds_df, df])
    odds_df.index = range(len(odds_df))
    odds = odds_df[['Home Odds Close', 'Away Odds Close']].max(axis=1).values    
    
    # Initialize all necessary variables & data structures for computations
    portfolio = pd.DataFrame({'Actuals': truths, 'Preds': probs, 'Odds': odds})
    sum_bets, sum_winnings = 0, 0
    
    # Create df where predict_proba > threshold value (greater confidence of underdog winning)
    win_prob = portfolio['Preds'] > thresh
    good_bets = portfolio[win_prob]
    
    # Iterate over every row
    for index, row in good_bets.iterrows():
        # Track the total amount of money placed on bets
        sum_bets += stake

        if row['Actuals'] == 1:
            # Track the total amount of earnings from winning bets
            sum_winnings += (stake * row['Odds'])
        else:
            # If the underdog loses, the loss is already accounted for by tracking how much money was bet
            # and the fact that no money was won (comes into play during ROI colculation)
            continue
    
    # ROI calculation
    roi = (sum_winnings - sum_bets) / sum_bets

    return roi







def logistic_bakeoff(yearlist, scoremethod):
    '''
    Function that will perform gridsearch on logistic regression for specific 
    year range and output ROIs for all rolling windows over that period based 
    on 2018-2019/2019-2020 season betting portfolio
    
    yearlist: type list of ints from 2009-2019
    scoremethod: any metric of interest for gridsearch (f1, precision, recall, etc)
    '''
    
    train_dfs, test_dfs = {}, {}
    for i in range(5,51):
        train_dfs[f'train_{i}'] = construct_df(yearlist, i, os.getcwd()+'\\')
        test_dfs[f'test_{i}'] = construct_df(range(2019,2021), i, os.getcwd()+'\\')
    
    # Generate training and testing datasets for all rolling windows
    X_trains, Y_trains, X_tests, Y_tests = {}, {}, {}, {}
    for i in train_dfs:
        X_trains[i] = train_dfs[i].drop('Underdog Win', 1)
        Y_trains[i] = train_dfs[i]['Underdog Win']
    for i in test_dfs:
        X_tests[i] = test_dfs[i].drop('Underdog Win', 1)
        Y_tests[i] = test_dfs[i]['Underdog Win']
        
    #Use Kfold to create 5 folds
    kfolds = KFold(n_splits = 5)
    
    #Create a set of steps. All but the last step is a transformer (something that processes data). 
    #Build a list of steps, where the first is StandardScaler and the last is LogisticRegression
    #PCA questionable here, but improves runtime significantly
    steps = [('scaler', StandardScaler()),
             ('pca', PCA()),
             ('lr', LogisticRegression(solver = 'liblinear'))]
    
    #Now set up the pipeline
    pipeline = Pipeline(steps)
    
    #Now set up the parameter grid
    parameters_scaler = dict(lr__C = [10**i for i in range(-3, 3)],
                             lr__class_weight = ['balanced', None],
                             lr__penalty = ['l1', 'l2'],
                             lr__fit_intercept = [True, False],
                             lr__max_iter = [100, 1000, 10000])
    
    #Now run a grid search
    lr_grid_search_scaler = GridSearchCV(pipeline, param_grid = parameters_scaler, cv = kfolds, scoring = scoremethod)
    
    #Generate various metrics for all rolling windows and store in dictionaries
    predicts, predict_probas, precisions, recalls, f1s, aucs, accuracies = {}, {}, {}, {}, {}, {}, {}
    for i, j in list(zip(X_trains, X_tests)):
        lr_grid_search_scaler.fit(X_trains[i], Y_trains[i])
        predicts[j] = lr_grid_search_scaler.predict(X_tests[j])
        predict_probas[j] = lr_grid_search_scaler.predict_proba(X_tests[j])
        precisions[j] = precision_score(Y_tests[j], predicts[j])
        recalls[j] = recall_score(Y_tests[j], predicts[j])
        f1s[j] = f1_score(Y_tests[j], predicts[j])
        aucs[j] = roc_auc_score(Y_tests[j], predicts[j])
        accuracies[j] = accuracy_score(Y_tests[j], predicts[j])
    
    #Generate list of ROIs to determine rolling window to use - can set multiple thresh values
    rois = {}
    for i in predict_probas:
        rois[i] = portfolio_roi(Y_tests[i], predict_probas[i][:,1], 10)




