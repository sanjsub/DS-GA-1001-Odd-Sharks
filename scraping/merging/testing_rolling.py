#%%
import sys
# sys.path.append("..\DS-GA-1001-Odd-Sharks\scraping\merging\cleaned_dfs_11-23")
# testing for linux
sys.path.append("cleaned_dfs_11-23/")
from rollstats_and_oddsfilter import roll_stats
import pandas as pd
years = list(range(2009, 2010))
#year = 2009

#bookie_df = pd.read_csv('..\DS-GA-1001-Odd-Sharks\scraping\merging\cleaned_dfs_11-23\{}_stats.csv'.format(year), parse_dates=['DATE'])
# %%
lst = []
for year in years:
    bookie_df = pd.read_csv('cleaned_dfs_11-23/{}_stats.csv'.format(year), parse_dates=['DATE'])
    # bookie_df = pd.read_csv('..\DS-GA-1001-Odd-Sharks\scraping\merging\cleaned_dfs_11-23\{}_stats.csv'.format(year), parse_dates=['DATE'])
    lst.append(roll_stats(bookie_df, n=10))
# %%
lst[0].to_csv("tester.csv")
# %%
