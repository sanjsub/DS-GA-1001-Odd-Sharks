import pandas as pd
import numpy as np

def underdogs_faves_aggregrates(bookie_df):
    # Create a unique game ID to identify games on which ot perform groupby operation
    bookie_df['unique_game_id'] = bookie_df['DATE'].astype(str) + '_' + bookie_df['Home Team'] + '_' + bookie_df['Away Team']    
    main_game_data = bookie_df['unique_game_id'].drop_duplicates().str.split('_', expand=True)
    
    # Perform groupby to get mean bookie odds per game
    bookie_means = bookie_df.groupby(['unique_game_id']).mean()

    # Add unique game data back onto new df
    bookie_means.insert(1, 'DATE', list(pd.to_datetime(main_game_data[0])))
    bookie_means.insert(2, 'Home Team', list(main_game_data[1]))
    bookie_means.insert(3, 'Away Team', list(main_game_data[2]))
    bookie_means.insert(0, 'unique_game_id', list(bookie_means.index))
    
    # Filter odds based on significant underdogs & favorites
    odds_filter = ((bookie_means['Home Odds Close'].between(1, 1.5)) | (bookie_means['Home Odds Close'] >= 3) | 
                   (bookie_means['Away Odds Close'].between(1, 1.5)) | (bookie_means['Away Odds Close'] >= 3))
    underdogs_faves = bookie_means[odds_filter]
    
    # Feature cleanup
    underdogs_faves.drop(['Unnamed: 0', 'Bookie Number'], axis=1, inplace=True)
    underdogs_faves.index = range(len(underdogs_faves))
    
    return underdogs_faves

# Read in relevant CSV files
filepath = '../final merged csvs/'
bookie_df = pd.read_csv(filepath + '2019_stats.csv', parse_dates=['DATE'])

# Output heavy underdog/favorites df
underdogs_faves_aggregrates(bookie_df)