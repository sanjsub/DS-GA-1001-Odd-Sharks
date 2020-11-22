import numpy as np


def more_cleaning(schedule):
	schedule.reset_index(inplace=True)
	schedule["DATE"] = [date.strftime('%Y-%m-%d') for date in schedule["DATE"]]
	schedule["VISITOR_PTS"] = schedule["VISITOR_PTS"].astype(int)
	schedule["HOME_PTS"] = schedule["HOME_PTS"].astype(int)
	schedule["HOME_WIN"] = schedule["HOME_PTS"] > schedule["VISITOR_PTS"]
	schedule["WINNING_TEAM"] = schedule["VISITOR"]
	schedule["WINNING_TEAM"][schedule["HOME_WIN"] == True] = schedule["HOME"][schedule["HOME_WIN"] == True]


def games_team_played(schedule, team):
    return schedule[(schedule["HOME"] == team) | (schedule["VISITOR"] == team)]


def get_cum_win_pct(schedule, team, date):
    prior_games = schedule[schedule["DATE"] < date]
    prior_games_team = games_team_played(prior_games, team)
    if len(prior_games_team) > 0:
        return np.sum(prior_games_team["WINNING_TEAM"] == team) / len(prior_games_team)
    else:
        return 0.5


def append_cum_win_pct(schedule):
	schedule["HOME_CUM_WIN_PCT"]= 0
	schedule["AWAY_CUM_WIN_PCT"]= 0
	for index, row in schedule.iterrows():
		schedule["HOME_CUM_WIN_PCT"].iloc[[index]] = get_cum_win_pct(schedule, row["HOME"], row["DATE"])
		schedule["AWAY_CUM_WIN_PCT"].iloc[[index]] = get_cum_win_pct(schedule, row["VISITOR"], row["DATE"])


def get_rolling_win_pct(schedule, team, date, n_games):
    prior_games = schedule[schedule["DATE"] < date]
    prior_games_team = games_team_played(prior_games, team)
    if len(prior_games_team) > n_games:
        return np.sum(prior_games_team["WINNING_TEAM"][-n_games:] == team) / n_games
    elif len(prior_games_team) > 0:
        return np.sum(prior_games_team["WINNING_TEAM"] == team) / len(prior_games_team)
    else:
        return 0.5


def append_rolling_win_pct(schedule, n_games):
    schedule["HOME_ROLL_WIN_PCT"]= 0
    schedule["AWAY_ROLL_WIN_PCT"]= 0
    for index, row in schedule.iterrows():
        schedule["HOME_ROLL_WIN_PCT"].iloc[[index]] = get_rolling_win_pct(schedule, row["HOME"], row["DATE"], n_games)
        schedule["AWAY_ROLL_WIN_PCT"].iloc[[index]] = get_rolling_win_pct(schedule, row["VISITOR"], row["DATE"], n_games)

