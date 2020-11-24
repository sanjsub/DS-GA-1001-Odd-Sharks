#%%
import sys
# sys.path.append("..\DS-GA-1001-Odd-Sharks\scraping\merging\cleaned_dfs_11-23")
# testing for linux
sys.path.append("cleaned_dfs_11-23/")
from rollstats_and_oddsfilter import roll_stats
import pandas as pd
years = list(range(2019, 2020))
#year = 2009

#bookie_df = pd.read_csv('..\DS-GA-1001-Odd-Sharks\scraping\merging\cleaned_dfs_11-23\{}_stats.csv'.format(year), parse_dates=['DATE'])
# %%
for rolling_window in range(5, 6):    
    all_seasons = []
    for year in years:
        bookie_df = pd.read_csv('cleaned_dfs_11-23/{}_stats.csv'.format(year), parse_dates=['DATE'])
        # bookie_df = pd.read_csv('..\DS-GA-1001-Odd-Sharks\scraping\merging\cleaned_dfs_11-23\{}_stats.csv'.format(year), parse_dates=['DATE'])
        all_seasons.append(roll_stats(bookie_df, n=rolling_window))

    for model, year in zip(all_seasons, years):
        output_path = 'cleaned_dfs_11-23/all_rolling_windows/'+ str(year) + "_stats" + "_n"+ str(rolling_window) + ".csv"
        model.to_csv(output_path, index=False)
# %%
