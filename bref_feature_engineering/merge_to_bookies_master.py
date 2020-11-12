from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import copy

# from allstats import allstatsrolling
from bref_feature_engineering.players_all import get_vorps, get_players, add_player_features
from bref_feature_engineering.players_impact_star import get_impact_players, get_all_stars
from bref_feature_engineering.schedule import get_clean_schedule, merge_sched_to_bookies


def add_underdog_data(bookie_df):
    # Handle missing odds data
    bookie_df[bookie_df['Home Odds Close'] == 'NO DATA ODDS'] = float('NaN')
    bookie_df[bookie_df['Away Odds Close'] == 'NO DATA ODDS'] = float('NaN')

    # Binary feature to determine if the away team won
    bookie_df['Away Win'] = ((bookie_df['VISITOR_PTS'].astype(int) - 
                              bookie_df['HOME_PTS'].astype(int)) > 0).astype(int)
    
    # Binary feature to determine if the away team is an underdog by comparing the values of the opening odds
    bookie_df['Away Underdog'] = (bookie_df['Away Odds Close'].astype(float) > 
                                          bookie_df['Home Odds Close'].astype(float)).astype(int)
    
    # Binary feature to determine if the underdog won
    bookie_df['Underdog Win'] = (bookie_df['Away Underdog'] == bookie_df['Away Win']).astype(int)
    
    # Drop features that are irrelevant or may cause data leakage
    bookie_df.drop(['HOME', 'VISITOR', 'HOME_PTS', 'VISITOR_PTS', 'Game Date', 'D-1', 'D+1'],
                      axis=1, inplace=True)
    
    return bookie_df


"""-----------------------------------------------"""
"""Function calls"""

# Read in all relevant CSV files
filepath = '../scraping/'
bookie_odds = {}

for season in range(2009, 2021):
    bookie_odds[season] = pd.read_csv(filepath + '{}_season.csv'.format(season), parse_dates=['Game Date'])

# Get NBA player VORPs
vorps = get_vorps(2020)

# Get all NBA players
players = get_players(2020, vorps)

# Get impactful NBA players
impact_players = get_impact_players(players)

# Get likely all-stars
all_stars_count = get_all_stars(players)

# Create copy of bookies df for computations
bookie_df = bookie_odds[2020]

# Add player-specific features to our bookies df
bookies_and_players = add_player_features(bookie_df, players, all_stars_count)

# Get NBA's entire schedule
sched = get_clean_schedule(2020)

# Merge all relevant data into bookies df
bookie_sched_merge = merge_sched_to_bookies(bookies_and_players, sched)

# Add underdog data to bookies df, clean up features
clean_bookie_df = add_underdog_data(bookie_sched_merge)

clean_bookie_df.head()


"""Sanjay's function; incorporate in the future"""
# # Read in all relevant CSV files
# # Output final cleaned merged dataframes for each season

# def make_clean_dfs():
#     filepath = os.getcwd() + '\\csv odds files\\'
#     bookie_odds, players, clean_bookie_dfs = {}, {}, {}
    
#     for season in range(2009, 2021):
#         bookie_odds[season] = pd.read_csv(filepath + '{}_season.csv'.format(season), parse_dates = ['Game Date'])
#         bookie_odds[season] = bookie_odds[season].dropna(axis = 0, how = 'all')
#         players[season] = pd.read_csv(filepath + '{}_players.csv'.format(season))
    
#         # Get likely all-stars
#         all_stars_count = get_all_stars(players[season])
        
#         # Create copy of bookies df for computations
#         bookie_df = copy.deepcopy(bookie_odds[season])
        
#         # Add player-specific features to our bookies df
#         bookies_and_players = add_player_features(bookie_df, players[season], all_stars_count)
        
#         # Get NBA's entire schedule
#         sched = get_clean_schedule(season)
        
#         # Merge all relevant data into bookies df
#         bookie_sched_merge = merge_sched_to_bookies(bookies_and_players, sched)
        
#         # Add underdog data to bookies df, clean up features
#         clean_bookie_dfs[season] = add_underdog_data(bookie_sched_merge)
#     return clean_bookie_dfs

# clean_bookie_dfs = make_clean_dfs()