import pandas as pd
import numpy as np
from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats
from basketball_reference_scraper.players import get_stats, get_game_logs
from basketball_reference_scraper.seasons import get_schedule, get_standings
from basketball_reference_scraper.box_scores import get_box_scores
from API_fixes.getgamelogsfix import get_game_logs_fix


# test game log read-in on Kuzma, can ignore
#gamelog = get_game_logs_fix('Kyle Kuzma', '2018-10-01', '2019-04-30', playoffs=False)
# pulling names of LAL 2018-2019 roster, can ignore
#names = get_roster('LAL', 2019).PLAYER

# pulling relevant counting stats to use as features
#stats = gamelog.columns[9:28]
stats = np.array(['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GAME_SCORE'])


# returns dataframe of rolling averages for each player on a team for a specific date range
def rollingavgs(team, year, start_date, end_date):
    names = get_roster(team, year).PLAYER
    # some players have two-way contract marker (TW) - removing that
    for n in names:
        n = n.strip(' (TW)')
        
    # counting stats as rows and players as columns - to store rolling avgs
    rollingavg = pd.DataFrame(index = stats, columns = names)
    
    # pull game log for each player, remove irrelevant columns, convert to numeric, average, and put in rollingavg DF
    for n in names:
        gamelog = get_game_logs_fix(n, start_date, end_date, playoffs=False)
        gamelog = gamelog.drop(['DATE','AGE','TEAM','HOME/AWAY','OPPONENT','RESULT','GS','MP'], axis = 1).apply(pd.to_numeric).mean(axis = 0)
        rollingavg[n] = gamelog
        
    # add column of team rolling average for each stat    
    rollingavg['Team Average'] = rollingavg.mean(axis = 1)  
    # drop NaN columns (players without an appearance over that stretch of games)    
    return rollingavg.dropna(axis = 1, how = 'all')

# should output stats for 20 players who had playing time over this stretch plus team rolling average
#rolling = rollingavgs('LAL', 2019, '2019-02-20', '2019-02-25')



# returns dictionary of game logs for every player on a team over entire 82-game season
# will take a long ass time to run
def seasonstats(team, year, start_date = '2018-10-01', end_date = '2019-04-30'):
    names = get_roster(team, year).PLAYER
    for n in names:
        n = n.strip(' (TW)')
    
    # dictionary with game #s as keys
    seasonlog = dict.fromkeys(range(1,83), [])
    # for each game, pull game log for each player and add it to gamestats DF
    # gamestats will reset each loop and become input for the next game's stats
    # each key in seasonlog will end up with a dataframe of player stats for that game
    for s in seasonlog: 
        gamestats = pd.DataFrame(index = stats, columns = names)
        for n in names:
            gamelog = get_game_logs_fix(n, start_date, end_date, playoffs=False)
            gamelog = gamelog.drop(['DATE','AGE','TEAM','HOME/AWAY','OPPONENT','RESULT','GS','MP'], axis = 1).apply(pd.to_numeric)
            for g in gamelog.index:
                if s == g:
                    gamestats[n] = gamelog.loc[g]
        # printing each loop to make sure it works - should print the last player in 'names' 82 times
        print(n)
        seasonlog[s] = gamestats
    return seasonlog

# should give game by game stats for everyone who had playing time on Lakers in 2018-2019 season
# NaN values if they did not play in that game
#seasonlog = seasonstats('LAL', 2019)



# returns dataframe of box score results for a specific team in a given year
# Outputs margin of victory or defeat (GAMESCORE) along with game RESULT (W or L)
def teamrecord(team, year):
    sched = get_schedule(year, playoffs=False)
    sched = sched[(sched['VISITOR'] == team) | (sched['HOME'] == team)]
    sched = sched.values.tolist()
    for s in sched:
        s[0] = s[0].strftime('%Y-%m-%d')
        if s[1] == team:
            s.append(int(s[2]) - int(s[4]))
        elif s[3] == team:
            s.append(int(s[4]) - int(s[2]))
        if s[5] > 0:
            s.append(1)
        elif s[5] < 0:
            s.append(0)
    sched = pd.DataFrame(sched, columns = ['DATE','VISITOR','VISITOR_PTS','HOME','HOME_PTS','GAMESCORE','RESULT'])
    sched.set_index(sched.index + 1, inplace=True)
    return sched

# should give 82x7 dataframe for Celtics 2017-2018 season
#s = teamrecord('Boston Celtics', 2018)
