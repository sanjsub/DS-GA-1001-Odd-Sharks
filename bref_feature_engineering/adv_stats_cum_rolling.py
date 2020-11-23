import pandas as pd
import numpy as np

# Read in all relevant CSV files
bookie_df = pd.read_csv('DS-GA-1001-Odd-Sharks/final_merged_csvs_v2/2019_stats.csv', parse_dates=['DATE'])
bookie_df

# Select team for experimentation
team = 'Boston Celtics'


# Create df of all advanced stats from all home games of the given team
home_games = (bookie_df['Home Team'] == team)
home_season = bookie_df[home_games].drop_duplicates(['DATE', 'Home Team', 'Away Team'])

# Create list of all advanced stats features
adv_stats_cols = [col for col in home_season.columns if 'Home' in col]
del adv_stats_cols[:5]
adv_stats_cols.insert(0, 'DATE')

# Create df of advanced stats features only
home_stats = home_season[adv_stats_cols]

mod_adv_stats_cols = [col[5:] for col in adv_stats_cols if 'Home' in col]
mod_adv_stats_cols.insert(0, 'DATE')

home_stats.columns = mod_adv_stats_cols
home_stats


# Create df of all advanced stats from all away games of the given team
away_games = (bookie_df['Away Team'] == team)
away_season = bookie_df[away_games].drop_duplicates(['DATE', 'Home Team', 'Away Team'])

# Use same columns for away_stats as were used for the home_stats df
away_stats = away_season[adv_stats_cols]

away_stats.columns = mod_adv_stats_cols
away_stats


# Concatenate home_stats & away_stats to get a team's whole season worth of advanced stats
season = pd.concat([home_stats, away_stats])

# Keep track of which games are home and away
locations = (['H'] * len(home_stats)) + (['A'] * len(away_stats))
season['Location'] = locations

# Sort team's games by date
season.sort_values('DATE', inplace=True)
season.head()