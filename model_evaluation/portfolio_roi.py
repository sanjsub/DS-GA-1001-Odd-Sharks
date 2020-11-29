import numpy as np
import pandas as pd
from model_feature_selection.basic_feature_analysis_pi import construct_df

"""Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **stake - stake to bet on each game
    **thresh- confidence threshold for filtering bets

    returns an expected portfolio ROI
"""
def portfolio_roi(truths, probs, stake, thresh=0.5):
    # Get absolute odds from within function instead of passing them in as a parameter
    odds_df = pd.DataFrame()
    for year in [2019, 2020]:
        df = pd.read_csv(f'DS-GA-1001-Odd-Sharks/scraping/merging/cleaned_dfs_11-23/all_rolling_windows/{year}_stats_n5.csv')
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

"""--------------------------------------------"""
"""FUNCTION CALLS BELOW"""
"""--------------------------------------------"""

# Create df of our test set
test_df = construct_df([2019, 2020], 5, 
                       'DS-GA-1001-Odd-Sharks/scraping/merging/cleaned_dfs_11-23/all_rolling_windows/')
test_df.index = range(len(test_df))

# Truths; extract target variable from test_df
y = test_df['Underdog Win']

# Predictions; randomly generated "predict_proba" array
np.random.seed(1234)
probs = np.random.random((len(test_df)))

# Return and print portfolio ROI calculation
roi = portfolio_roi(y, probs, 100)
print(f'{np.round(roi * 100, 2)}%')