import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from model_feature_selection.basic_feature_analysis_pi import construct_df

"""Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **stake - stake to bet on each game
    **thresh- confidence threshold for filtering bets

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
        # Weight each bet based on higher risk games
        high_stake = stake * row['Odds']/2.0
        
        # Choose appropriate stake amount based on odds value
        select_stake = high_stake if (row['Odds'] >= 3) else stake
        
        # Track the total amount of money placed on bets
        sum_bets += select_stake

        if row['Actuals'] == 1:
            # Track the total amount of earnings from winning bets
            sum_winnings += (select_stake * row['Odds'])
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


"""Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **threshs - single confidence threshold or list/array of confidence thresholds

    returns a dataframe with the resulting ROI, precision, & recall given a threshold;
    dataframe indexed by threshold value
"""
def test_thresholds(truths, probs, threshs, main_path='DS-GA-1001-Odd-Sharks/scraping/merging/cleaned_dfs_11-23/all_rolling_windows/'):
    # Initialize threshold stats df
    thresh_stats = pd.DataFrame(columns=['ROI', 'Precision', 'Recall'])
    
    # Ensure threshs are list for iteration purposes
    if type(threshs) in [list, np.ndarray]:
        for thresh in threshs:
            # Calculate game outcome binary predictions based on (probability estimate > threshold)
            preds = [1 if (num > thresh) else 0 for num in probs]

            # Calculate ROI given threshold
            history, roi = portfolio_roi(truths, probs, 100, thresh, main_path)

            # Calculate precision & recall of predictions
            precision = precision_score(truths, preds)
            recall = recall_score(truths, preds)

            # Store stats in df; indexed by threshold
            thresh_stats.loc[thresh] = [roi, precision, recall]
    
    # Same process as above, only with a number instead of list/array
    else:
        preds = [1 if (num > threshs) else 0 for num in probs]

        roi = portfolio_roi(truths, probs, 100, threshs, main_path)

        precision = precision_score(truths, preds)
        recall = recall_score(truths, preds)

        thresh_stats.loc[threshs] = [roi, precision, recall]
        
    # Round off stats to 4 decimal places to make df more readable
    for col in thresh_stats.columns:
        thresh_stats[col] = np.round(thresh_stats[col], 4)
        
    return thresh_stats

"""--------------------------------------------"""
"""FUNCTION CALLS BELOW"""
"""--------------------------------------------"""

# # Create df of our test set
# test_df = construct_df([2019, 2020], 5, 
#                        'DS-GA-1001-Odd-Sharks/scraping/merging/cleaned_dfs_11-23/all_rolling_windows/')
# test_df.index = range(len(test_df))

# # Truths; extract target variable from test_df
# y = test_df['Underdog Win']

# # Predictions; randomly generated "predict_proba" array
# np.random.seed(1234)
# probs = np.random.random((len(test_df)))

# # Get bet-by-bet portfolio history & final portfolio ROI
# history, roi = portfolio_roi(y, probs, 100,)
# print(f'Portfolio ROI: {roi}')
# plt.plot(range(1, len(history) + 1), history)

# # Get the corresponding ROI, precision, & recall for given threshold value(s)
# test_thresholds(y, probs, np.arange(0.5, 1, 0.01))