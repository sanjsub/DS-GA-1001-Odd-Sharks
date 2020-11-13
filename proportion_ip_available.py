from getgamelogsfix import get_game_logs_fix
from players_all import get_vorps, get_players
from players_impact_star import get_impact_players


# assuming you have run season_series on a given year and have sched DF 
def proportion_ip_avail(sched):
    year = int(sched['DATE'].iloc[-1][0:4])
    vorps = get_vorps(year)
    players = get_players(year, vorps)
    impact = get_impact_players(players)
    
    team_logs = {}
    for team in impact.keys():
        player_logs = {}
        for player in impact[team]:
            print(player)
            player_logs[player] = get_game_logs_fix(player, sched['DATE'].iloc[0], sched['DATE'].iloc[-1])
        team_logs[team] = player_logs
    
    sched['HOME_IP_AVAIL'] = ''
    sched['AWAY_IP_AVAIL'] = ''
    
    for idx, row in sched.iterrows():
        date = sched['DATE'].iloc[idx]
        home = sched['HOME'].iloc[idx] 
        away = sched['VISITOR'].iloc[idx]
        home_count, away_count = 0, 0
        for player in team_logs[home]:
            for d in team_logs[home][player]['DATE']:
                if date == d:
                    home_count += 1
        for player in team_logs[away]:
            for d in team_logs[away][player]['DATE']:
                if date == d:
                    away_count += 1
        sched['HOME_IP_AVAIL'].iloc[idx] = home_count / 7
        sched['AWAY_IP_AVAIL'].iloc[idx] = away_count / 7
    return sched




