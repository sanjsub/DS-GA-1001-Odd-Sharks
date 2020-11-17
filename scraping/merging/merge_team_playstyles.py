#%%
import pandas as pd
import numpy as np
import os

## ADD ANY TEAM ABBREVIATION TO THIS DICT AND OUR MAPPED DICTIONARIES WILL BE GOOD
team_name_to_num = {
'Atlanta Hawks':1,
'Boston Celtics':2,
'Brooklyn Nets':3,
'New Jersey Nets':3,
'Charlotte Bobcats':4,
'Charlotte Hornets':4,
'Chicago Bulls':5,
'Cleveland Cavaliers':6,
'Dallas Mavericks':7,
'Denver Nuggets':8,
'Detroit Pistons':9,
'Golden State Warriors':10,
'Houston Rockets':11,
'Indiana Pacers':12,
'LA Clippers':13,
'Los Angeles Clippers':13,
'Los Angeles Lakers':14,
'Memphis Grizzlies':15,
'Miami Heat':16,
'Milwaukee Bucks':17,
'Minnesota Timberwolves':18,
'New Orleans Hornets':19,
'New Orleans Pelicans':19,
'New York Knicks':20,
'Oklahoma City Thunder':21,
'Orlando Magic':22,
'Philadelphia 76ers':23,
'Phoenix Suns':24,
'Portland Trail Blazers':25,
'Sacramento Kings':26,
'San Antonio Spurs':27,
'Toronto Raptors':28,
'Utah Jazz':29,
'Washington Wizards':30
}

num_to_team_name = {i:[] for i in range(1,31)}

for key, value in team_name_to_num.items():
    num_to_team_name[value].append(key)

# %%
current_master_df = pd.read_csv("2015_bookie_merge.csv")
current_master_df['Home Team Num'] = current_master_df['Home Team'].apply(lambda x: team_name_to_num[x])
current_master_df['Away Team Num'] = current_master_df['Away Team'].apply(lambda x: team_name_to_num[x])
# %%

# time to merge
year = '2015'
filename = 'Cut_offensive.xlsx'
path = 'team_playtype_stats/' + filename

team_play_stats = pd.read_excel(path, sheet_name=year)[['TEAM', 'PERCENTILE']].loc[1:]
team_play_stats['Team Num'] = team_play_stats['TEAM'].apply(lambda x: team_name_to_num[x])
percentiles = {row['Team Num']:row['PERCENTILE']/100 for row_num, row in team_play_stats.iterrows()}

stat = filename[:-5]
#%%
current_master_df['Home team ' + stat] = current_master_df['Home Team Num'].apply(lambda x: percentiles[x])


#%%
## Let's try all stats now
all_files = os.listdir('team_playtype_stats/')
all_files.remove('Archive')
year = '2015'

for filename in all_files:
    path = 'team_playtype_stats/' + filename

    team_play_stats = pd.read_excel(path, sheet_name=year)[['TEAM', 'PERCENTILE']].loc[1:]
    team_play_stats['Team Num'] = team_play_stats['TEAM'].apply(lambda x: team_name_to_num[x])
    percentiles = {row['Team Num']:row['PERCENTILE']/100 for row_num, row in team_play_stats.iterrows()}
    stat = filename[:-5]

    current_master_df['Home Team ' + stat] = current_master_df['Home Team Num'].apply(lambda x: percentiles[x])
    current_master_df['Away Team ' + stat] = current_master_df['Away Team Num'].apply(lambda x: percentiles[x])
