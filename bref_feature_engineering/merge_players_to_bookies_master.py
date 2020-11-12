from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from datetime import date

!pip install basketball-reference-scraper
from basketball_reference_scraper.teams import get_roster_stats

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


def get_schedule(season, playoffs=False):
    months = ['October', 'November', 'December', 'January', 'February', 'March',
            'April', 'May', 'June']
    if season==2020:
        months = ['October', 'November', 'December', 'January', 'February', 'March',
                'July', 'August', 'September', 'October']
    df = pd.DataFrame()
    for month in months:
        r = requests.get(f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month.lower()}.html')
        if r.status_code==200:
            soup = BeautifulSoup(r.content, 'html.parser')
            table = soup.find('table', attrs={'id': 'schedule'})
            month_df = pd.read_html(str(table))[0]
            df = df.append(month_df)
    df = df.reset_index()
    cols_to_remove = [i for i in df.columns if 'Unnamed' in i]
    cols_to_remove += [i for i in df.columns if 'Notes' in i]
    cols_to_remove += [i for i in df.columns if 'Start' in i]
    cols_to_remove += [i for i in df.columns if 'Attend' in i]
    cols_to_remove += ['index']
    df = df.drop(cols_to_remove, axis=1)
    df.columns = ['DATE', 'VISITOR', 'VISITOR_PTS', 'HOME', 'HOME_PTS']
    if season==2020:
        df = df[df['DATE']!='Playoffs']
        df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x))
        df = df.sort_values(by='DATE')
        df = df.reset_index().drop('index', axis=1)
        playoff_loc = df[df['DATE']==pd.to_datetime('2020-08-17')].head(n=1)
        if len(playoff_loc.index)>0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        if playoffs:
            df = df[playoff_index:]
        else:
            df = df[:playoff_index]
    else:
        playoff_loc = df[df['DATE']=='Playoffs']
        if len(playoff_loc.index)>0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        if playoffs:
            df = df[playoff_index+1:]
        else:
            df = df[:playoff_index]
        df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x))
    return df


def get_all_stars(players):
    # Create a dataframe of likely NBA all-stars based on 95% percentile of VORPs
    prod_threshold = np.percentile(players['VORP'], 95)
    all_stars = players[players['VORP'] >= prod_threshold]
    
    # Generate the distribution of likely NBA all-stars on each team
    all_stars_count = all_stars[['TEAM','VORP']].groupby('TEAM').count()

    for team in teams.values():
        if team not in all_stars_count.index:
            all_stars_count.loc[team] = {'VORP': 0}
    
    return all_stars_count


def get_clean_schedule(year):
    sched = get_schedule(year)

    sched.replace({'New Jersey Nets': 'Brooklyn Nets', 
                     'Charlotte Bobcats': 'Charlotte Hornets',
                     'New Orleans Hornets': 'New Orleans Pelicans',
                     'Seattle Supersonics': 'Oklahoma City Thunder'}, inplace=True)
    
    return sched


def add_player_features(bookie_df, players, all_stars_count):
    # Features indicating the number of all-stars on both the home & away teams
    bookie_df['Home All Stars'] = all_stars_count.loc[bookie_df['Home Team']].values
    bookie_df['Away All Stars'] = all_stars_count.loc[bookie_df['Away Team']].values
    
    # Calculate the median age of each NBA team
    players['AGE'] = players['AGE'].astype(int)
    med_team_age = players[['TEAM', 'AGE']].groupby('TEAM').median()

    bookie_df['Home Median Age'] = med_team_age.loc[bookie_df['Home Team']].values
    bookie_df['Away Median Age'] = med_team_age.loc[bookie_df['Away Team']].values
    
    return bookie_df


def merge_sched_to_bookies(bookie_df, sched):
    # Create extra potential game dates
    bookie_df['D-1'] = pd.to_datetime(bookie_df['Game Date']) + pd.Timedelta(days=-1)
    bookie_df['D+1'] = pd.to_datetime(bookie_df['Game Date']) + pd.Timedelta(days=1)
    
#     bookie_df['unique_games_test'] = bookie_df['Game Date'].astype(str) + bookie_df['Home Team'] + bookie_df['Away Team']    
    
    # Perform inner joins on all scheduled games with the possible dates created above
    merged_scores_d0 = sched.merge(bookie_df, left_on=['DATE', 'HOME', 'VISITOR'],
                                    right_on=['Game Date', 'Home Team', 'Away Team'],
                                    how='inner')

    merged_scores_dless1 = sched.merge(bookie_df, left_on=['DATE', 'HOME', 'VISITOR'],
                                    right_on=['D-1', 'Home Team', 'Away Team'],
                                    how='inner')
    
    merged_scores_dplus1 = sched.merge(bookie_df, left_on=['DATE', 'HOME', 'VISITOR'],
                                    right_on=['D+1', 'Home Team', 'Away Team'],
                                    how='inner')

    # Concatentate dataframes created above to generate an accurately dated bookies df
    frames = [merged_scores_d0, merged_scores_dless1, merged_scores_dplus1]
    merged_scores = pd.concat(frames, ignore_index=True)
    
    merged_scores.sort_values(by=['DATE', 'HOME'], inplace=True)
    
    merged_scores.drop_duplicates(inplace=True)

    return merged_scores


def add_underdog_data(bookie_df):
    # Binary feature to determine if the away team won
    bookie_df['Away Win'] = ((bookie_df['VISITOR_PTS'].astype(int) - 
                              bookie_df['HOME_PTS'].astype(int)) > 0).astype(int)
    
    # Binary feature to determine if the away team is an underdog by comparing the values of the opening odds
    bookie_df['Away Underdog'] = (bookie_df['Away Odds Close'].astype(float) > 
                                          bookie_df['Home Odds Close'].astype(float)).astype(int)
    
    # Binary feature to determine if the underdog won
    bookie_df['Underdog Win'] = (bookie_df['Away Underdog'] & bookie_df['Away Win']).astype(int)
    
    # Drop features that are irrelevant or may cause data leakage
    bookie_df.drop(['HOME', 'VISITOR', 'HOME_PTS', 'VISITOR_PTS', 'Game Date', 'D-1', 'D+1'],
                      axis=1, inplace=True)
    
    return bookie_df


"""-----------------------------------------------"""
"""Function calls"""

# Read in all relevant CSV files
filepath = '../scraping/'
bookie_odds, players = {}, {}

for season in range(2009, 2021):
    bookie_odds[season] = pd.read_csv(filepath + '{}_season.csv'.format(season), parse_dates=['Game Date'])
    players[season] = pd.read_csv(filepath + '{}_players.csv'.format(season))

# Get likely all-stars
all_stars_count = get_all_stars(players[2020])

# Create copy of bookies df for computations
bookie_df = bookie_odds[2020]

# Add player-specific features to our bookies df
bookies_and_players = add_player_features(bookie_df, players[2020], all_stars_count)

# Get NBA's entire schedule
sched = get_clean_schedule(2020)

# Merge all relevant data into bookies df
bookie_sched_merge = merge_sched_to_bookies(bookies_and_players, sched)

# Add underdog data to bookies df, clean up features
clean_bookie_df = add_underdog_data(bookie_sched_merge)

clean_bookie_df.head()