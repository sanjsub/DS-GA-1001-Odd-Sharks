import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from model_feature_selection.basic_feature_analysis_pi import construct_df

"""Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **stake - stake to bet on each game
    **thresh- confidence threshold for filtering bets
    **test_years - years from which to get test data
    **main_path - file access URL path

    returns 
    **bet_history - list of realized ROI following every bet placed
    **roi - expected portfolio ROI
"""
def portfolio_roi(truths, probs, stake, thresh=0.5, test_years=[2019, 2020],
                  main_path='DS-GA-1001-Odd-Sharks/scraping/merging/cleaned_dfs_11-23/all_rolling_windows/'):
    # Get absolute odds from within function instead of passing them in as a parameter
    odds_df = pd.DataFrame()
    for year in test_years:
        df = pd.read_csv(f'{main_path}{year}_stats_n5.csv')
        odds_df = pd.concat([odds_df, df])
    odds_df.index = range(len(odds_df))
    odds = odds_df[['Home Odds Close', 'Away Odds Close']].max(axis=1).values    
    
    # Initialize all necessary variables & data structures for computations
    portfolio = pd.DataFrame({'Actuals': truths, 'Preds': probs, 'Odds': odds})
    sum_bets, sum_winnings = 0, 0
    bet_history = []
    
    win_prob = portfolio['Preds'] >= thresh
    good_bets = portfolio[win_prob]
                
    # Iterate over every row
    for index, row in good_bets.iterrows():
        # # Weight each bet based on higher risk games
        # high_stake = stake * row['Odds']/2.0
        
        # # Choose appropriate stake amount based on odds value
        # select_stake = high_stake if (row['Odds'] >= 3) else stake
        
        # Track the total amount of money placed on bets
        sum_bets += stake

        if row['Actuals'] == 1:
            # Track the total amount of earnings from winning bets
            sum_winnings += (stake * row['Odds'])
        else:
            # If the underdog loses, the loss is already accounted for by tracking how much money was bet
            # and the fact that no money was won (comes into play during ROI colculation)
            pass
           
        # Append change to portfolio
        current_pct = ((sum_winnings - sum_bets) / sum_bets) * 100
        bet_history.append(current_pct)
    
    # ROI calculation
    if sum_bets == 0:
        roi = 0
    else:
        roi = (sum_winnings - sum_bets) / sum_bets

    return bet_history, roi


"""Function that plots a model's game-by-game ROI against that of random guessing
    Accepts the following arguments:
    **models - dict of models w/ their betting histories
    **randoms - betting history of random guessing
"""
def plot_vs_random(models, randoms):
    for model in models.keys():
        print('{} ROI: {:.2f}%'.format(model, models[model][-1]))

    print('Random ROI: {:.2f}%'.format(randoms[-1]))

    for model in models.keys():
        plt.plot(range(1, len(models[model]) + 1), models[model], label=f'{model}')
        
    plt.plot(range(1, len(randoms) + 1), randoms, label='Random Guessing')

    plt.title('Per Game Portfolio ROI')
    plt.xlabel('Games Bet On')
    plt.ylabel('ROI (%)')
    plt.legend()


"""Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **threshs - single confidence threshold or list/array of confidence thresholds
    **test_years - years from which to get test data
    **main_path - file access URL path

    returns a dataframe with the resulting ROI, precision, & recall given a threshold;
    dataframe indexed by threshold value
"""
def test_thresholds(truths, probs, threshs, test_years=[2019, 2020], 
                    main_path='DS-GA-1001-Odd-Sharks/scraping/merging/cleaned_dfs_11-23/all_rolling_windows/'):
    # Initialize threshold stats df
    thresh_stats = pd.DataFrame(columns=['ROI', 'Precision', 'Recall'])
    
    # Ensure threshs are list for iteration purposes
    if type(threshs) in [list, np.ndarray]:
        for thresh in threshs:
            # Calculate game outcome binary predictions based on (probability estimate > threshold)
            preds = [1 if (num > thresh) else 0 for num in probs]

            # Calculate ROI given threshold
            history, roi = portfolio_roi(truths, probs, 100, thresh, test_years, main_path)

            # Calculate precision & recall of predictions
            precision = precision_score(truths, preds)
            recall = recall_score(truths, preds)

            # Store stats in df; indexed by threshold
            thresh_stats.loc[thresh] = [roi, precision, recall]
    
    # Same process as above, only with a number instead of list/array
    else:
        preds = [1 if (num > threshs) else 0 for num in probs]

        roi = portfolio_roi(truths, probs, 100, threshs, test_years, main_path)

        precision = precision_score(truths, preds)
        recall = recall_score(truths, preds)

        thresh_stats.loc[threshs] = [roi, precision, recall]
        
    # Round off stats to 4 decimal places to make df more readable
    for col in thresh_stats.columns:
        thresh_stats[col] = np.round(thresh_stats[col], 4)
        
    return thresh_stats

"""--------------------------------------------"""
"""EXAMPLE FUNCTION CALLS"""
"""--------------------------------------------"""

# # Create df of our test set
# test_df = construct_df([2019, 2020], 5, 
#                        'DS-GA-1001-Odd-Sharks/scraping/merging/cleaned_dfs_11-23/all_rolling_windows/')
# test_df.index = range(len(test_df))

# # Truths; extract target variable from test_df
# y = test_df['Underdog Win']

# # Underdog win probabilities
# gbt_probs = pd.read_csv('gbt_proba_vec.csv', header=None)
# rf_probs = pd.read_csv('rf_proba_vec.csv', header=None)

# # Take probability estimation averages
# avg_gbt_probs = gbt_probs.mean(axis=1)
# avg_rf_probs = rf_probs.mean(axis=1)

# # Random guessing; based on 23% win probability of underdog
# hist_df = pd.DataFrame()
# min_hist = 1000

# for i in range(5000):
#     rand = np.random.choice([0, 1], size=(len(y),), p=[0.77, 0.23])
#     rand_history, rand_roi = portfolio_roi(y, rand, 10, test_years=test_years)
    
#     if len(rand_history) < min_hist:
#         min_hist = len(rand_history)
    
#     hist_df = pd.concat([hist_df, pd.Series(rand_history)], ignore_index=True, axis=1)

# # Average out Monte Carlo simulations
# avg_rand_history = hist_df.iloc[:min_hist].mean(axis=1).values

# # Model predictions
# rf_pred_history, rf_pred_roi = portfolio_roi(y, avg_rf_probs, 10, test_years=test_years)
# gbt_pred_history, gbt_pred_roi = portfolio_roi(y, avg_gbt_probs, 10, test_years=test_years)

# rf_pred_history.insert(0, 0)
# gbt_pred_history.insert(0, 0)

# models = {'Random Forest': rf_pred_history, 'Gradient-Boosted Tree': gbt_pred_history}

# # Compare betting strategies
# plot_vs_random(models, avg_rand_history)