from bs4 import BeautifulSoup
import requests
import datetime
import pandas as pd
import numpy as np
from datetime import date

!pip install basketball-reference-scraper
from basketball_reference_scraper.teams import get_roster_stats, get_team_stats, get_opp_stats
from basketball_reference_scraper.seasons import get_schedule, get_standings

teams = {
    "ATL":"Atlanta Hawks",
    "BRK":"Brooklyn Nets",
    "BOS":"Boston Celtics",
    "CHA":"Charlotee Bobcats",
    "CHO":"Charlotte Hornets",
    "CHI":"Chicago Bulls",
    "CLE":"Cleveland Cavaliers",
    "DAL":"Dallas Mavericks",
    "DEN":"Denver Nuggets",
    "DET":"Detroit Pistons",
    "GSW":"Golden State Warriors",
    "HOU":"Houston Rockets",
    "IND":"Indiana Pacers",
    "LAC":"Los Angeles Clippers",
    "LAL":"Los Angeles Lakers",
    "MEM":"Memphis Grizzlies",
    "MIA":"Miami Heat",
    "MIL":"Milwaukee Bucks",
    "MIN":"Minnesota Timberwolves",
    "NJN":"New Jersey Nets",
    "NOH":"New Orleans Hornets",
    "NOP":"New Orleans Pelicans",
    "NYK":"New York Knicks",
    "OKC":"Oklahoma City Thunder",
    "ORL":"Orlando Magic",
    "PHI":"Philadelphia 76ers",
    "PHO":"Phoenix Suns",
    "POR":"Portland Trail Blazers",
    "SAC":"Sacramento Kings",
    "SAS":"San Antonio Spurs",
    "SEA":"Seattle Supersonics",
    "TOR":"Toronto Raptors",
    "UTA":"Utah Jazz",
    "WAS":"Washington Wizards"
}


def get_player_stats(year, vorps):
    player_stats = pd.DataFrame()

    # Retrieve season stats for all active NBA players in a chosen year
    for team in teams.keys():
        player_stats = pd.concat([player_stats, get_roster_stats(team, year)], ignore_index=True)
        
    # Merge players' season stats with their respective VORPs
    all_stats = player_stats.merge(vorps, on='PLAYER').drop_duplicates('PLAYER')
    
    # Change team naming convention to match bookies CSVs
    full_names = []
    for abb in all_stats['TEAM']:
        full_names.append(teams[abb])

    all_stats['TEAM'] = full_names
    
    return all_stats


def get_all_stars(player_stats):
    # Create a dataframe of likely NBA all-stars based on 95% percentile of VORPs
    prod_threshold = np.percentile(player_stats['VORP'], 95)
    all_stars = player_stats[player_stats['VORP'] >= prod_threshold]
    
    # Generate the distribution of likely NBA all-stars on each team
    all_stars_count = all_stars[['TEAM','VORP']].groupby('TEAM').count()

    for team in teams.values():
        if team not in all_stars_count.index:
            all_stars_count.loc[team] = {'VORP': 0}
    
    return all_stars_count


"""Function to get VORP stats for all 
    active NBA players in a chosen year"""
def get_vorps(year):
    # Get rows of data
    page = requests.get('https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'.format(year))
    soup = BeautifulSoup(page.text, "html.parser")
    table = soup.find("table", attrs={"class":"sortable stats_table"})
    rows = table.find_all("tr")
    
    players, vorps = [], []

    # Iterate through each row to extract the player's name and their VORP
    for row in rows:
        player = row.find('a')
        vorp = row.find("td", attrs={"data-stat":"vorp"})

        # Only extract non-null data
        if player != None:
            players.append(player.text)
            vorps.append(vorp.text)

    data = {
        'PLAYER': players,
        'VORP': pd.Series(vorps).astype(float)
    }

    return pd.DataFrame(data=data).drop_duplicates('PLAYER')


def get_clean_schedule(year):
    sched = get_schedule(year)

    sched['DATE'] = sched['DATE'].astype(str)

    sched.replace({'New Jersey Nets': 'Brooklyn Nets', 
                     'Charlotte Bobcats': 'Charlotte Hornets',
                     'New Orleans Hornets': 'New Orleans Pelicans',
                     'Seattle Supersonics': 'Oklahoma City Thunder'}, inplace=True)

    sched['Away Win'] = ((sched['VISITOR_PTS'].astype(int) - 
                             sched['HOME_PTS'].astype(int)) > 0).astype(int)
    
    return sched


def add_player_features(season_df, player_stats, all_stars_count):
    # Features indicating the number of all-stars on both the home & away teams
    season_df['Home All Stars'] = all_stars_count.loc[season_df['Home Team']].values
    season_df['Away All Stars'] = all_stars_count.loc[season_df['Away Team']].values
    
    # Calculate the average age of each NBA team
    player_stats['AGE'] = player_stats['AGE'].astype(int)
    avg_team_age = player_stats[['TEAM', 'AGE']].groupby('TEAM').mean()

    season_df['Home Roster Age'] = avg_team_age.loc[season_df['Home Team']].values
    season_df['Away Roster Age'] = avg_team_age.loc[season_df['Away Team']].values

    # Binary feature to determine if the away team is an underdog by comparing the values of the opening odds
    season_df['Away Underdog'] = (season_df['Away Odds Open'].astype(float) > 
                                          season_df['Home Odds Open'].astype(float)).astype(int)
    
    return season_df


def merge_to_bookies(season_df, sched):
    merged_scores = season_df.merge(sched, left_on=['Game Date', 'Home Team', 'Away Team'],
                            right_on=['DATE', 'HOME', 'VISITOR'], how='left')

    merged_scores.drop(['DATE', 'HOME', 'VISITOR'], axis=1, inplace=True)

    merged_scores.rename(columns={'VISITOR_PTS': 'Away Score', 
                                 'HOME_PTS': 'Home Score'}, inplace=True)

#     merged_scores.dropna(subset=['Away Win', 'Away Underdog'], inplace=True)
    merged_scores['Underdog Win'] = (merged_scores['Away Underdog'] & merged_scores['Away Win']).astype(int)
    
    return merged_scores


"""-----------------------------------------------"""
"""Function calls"""

# Read in all odds CSV files
filepath = '../scraping/'
season_odds = {}

for season in range(2008, 2020):
    season_odds[season] = pd.read_csv(filepath + '{}_season.csv'.format(season))
    season_odds[season]['Game Date'] = (pd.to_datetime(season_odds[season]['Game Date']) + pd.Timedelta(days=-1)).astype(str)
    
season_odds[2019].head()

# Get all active NBA players' VORPs
vorps = get_vorps(2020)

# Get all active NBA players' stats
player_stats = get_player_stats(2020, vorps)

# Get likely all-stars
all_stars_count = get_all_stars(player_stats)

# Add player-specific features to our bookies dataframe
season_odds[2019] = add_player_features(season_odds[2019], player_stats, all_stars_count)

# Get NBA's entire schedule
sched = get_clean_schedule(2020)

# Merge all relevant data into bookies dataframe
merged_scores = merge_to_bookies(season_odds[2019], sched)
merged_scores.head()