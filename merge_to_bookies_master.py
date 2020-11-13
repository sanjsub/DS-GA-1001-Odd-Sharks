from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import copy
import os

# from allstats import allstatsrolling
from players_all import add_player_features
from players_impact_star import get_all_stars
from schedule import get_clean_schedule, merge_sched_to_bookies
from season_series import get_season_series
from proportion_ip_available import proportion_ip_avail


def add_underdog_data(bookie_df):
    # Handle missing odds data

    bookie_df[bookie_df['Home Odds Close'] == 'NO DATA ODDS'] = float('NaN')
    bookie_df[bookie_df['Away Odds Close'] == 'NO DATA ODDS'] = float('NaN')
    #bookie_df.dropna(axis = 0, inplace = True, how = 'all')
    
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



"""Sanjay's function; incorporate in the future"""
# # Read in all relevant CSV files
# # Output final cleaned merged dataframes for each season

def make_clean_dfs():
     filepath = os.getcwd() + '\\csv odds files\\'
     bookie_odds, players, clean_bookie_dfs = {}, {}, {}
    
     for season in range(2019, 2020):
         bookie_odds[season] = pd.read_csv(filepath + '{}_season.csv'.format(season), parse_dates = ['Game Date'])
         bookie_odds[season] = bookie_odds[season].dropna(axis = 0, how = 'all')
         players[season] = pd.read_csv(filepath + '{}_players.csv'.format(season))
    
         # Get likely all-stars
         all_stars_count = get_all_stars(players[season])
        
         # Create copy of bookies df for computations
         bookie_df = copy.deepcopy(bookie_odds[season])
        
         # Add player-specific features to our bookies df
         bookies_and_players = add_player_features(bookie_df, players[season], all_stars_count)
        
         # Get NBA's entire schedule
         sched = get_clean_schedule(season)
         
         # Merge season series and proportion impact players to bookies df
         sched = get_season_series(sched)
         sched = proportion_ip_avail(sched)
        
         # Merge all relevant data into bookies df
         bookie_sched_merge = merge_sched_to_bookies(bookies_and_players, sched)
        
         bookie_sched_merge_test = copy.deepcopy(bookie_sched_merge)
         # Add underdog data to bookies df, clean up features
         clean_bookie_dfs[season] = add_underdog_data(bookie_sched_merge)
     return clean_bookie_dfs

clean_bookie_dfs = make_clean_dfs()

bookie_sched_merge.to_csv(os.getcwd()+'\\2019_bookie_merge.csv')

