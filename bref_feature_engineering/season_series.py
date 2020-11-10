import numpy as np
from bref_feature_engineering.rollingavg_seasonlog_teamrecord import teamrecord

def get_season_series(team_record, team_name, opponent_name, date):
	"""
	Returns the season series for team_name against opponent_name as of date.
	:param team_record: a dataframe consisting of all games played by a team 
		and their outcomes.
	:param team_name: the name of the team whose record is stored in team_record.
	:param opponent_name: the name of the opponent against which to calculate the 
		season series.
	:param date: the date as of which to calculate the season series.
	:return: tuple(number of wins against opponent, number of games against opponent).
	"""
	opponent_filter = (team_record['VISITOR'] == opponent_name) | (team_record['HOME'] == opponent_name)
	team_vs_opponent_prior = team_record[(team_record["DATE"] < date) & opponent_filter]
	WIN = 1
	return (np.sum(team_vs_opponent_prior["RESULT"] == WIN), len(team_vs_opponent_prior))


def append_season_series(team_record, team_name):
	"""
	Updates the dataframe team_record in place to include two new columns: 
		PRIOR_WINS_V_OPP and PRIOR_GAMES_V_OPP.
	Suppose team_record is a list of games played by the Boston Celtics.
	If a row contains Boston Celtics as VISITOR and Chicago Bulls as HOME, 
	then PRIOR_WINS_V_OPP in that row contains a count of the number of times
	the Celtics have beaten the Bulls previously during the season.
	PRIOR_GAMESS_V_OPP contains count of the nuber of times the Celtics played
	the Bulls previously during the season.
	
	:param team_record: a dataframe consisting of all games played by a team and
		 their outcomes.
	:param team_name: the name of the team whose record is stored in team_record.
	:return: None.

	Usage:
	s = teamrecord('Boston Celtics', 2018)
	append_season_series(s, "Boston Celtics")
	"""
	team_record["PRIOR_WINS_V_OPP"] = 0
	team_record["PRIOR_GAMES_V_OPP"] = 0
	for idx, row in team_record.iterrows():
		if row["VISITOR"] == team_name:
			opponent_name = row["HOME"]
		else:
			opponent_name = row["VISITOR"]
		prior_wins_v_opp, prior_games_v_opp = get_season_series(team_record, team_name, opponent_name, row["DATE"])
		team_record["PRIOR_WINS_V_OPP"].iloc[[idx]] = prior_wins_v_opp
		team_record["PRIOR_GAMES_V_OPP"].iloc[[idx]] = prior_games_v_opp



