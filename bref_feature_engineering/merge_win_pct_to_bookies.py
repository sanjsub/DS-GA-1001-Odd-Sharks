import pandas as pd


for year in range(2009, 2021):
	sched = pd.read_csv("../schedule_csvs/{}_schedule.csv".format(year))
	stats = pd.read_csv("../final merged csvs/{}_stats.csv".format(year))
	sched.rename(columns={"HOME": "Home Team", "VISITOR": "Away Team"}, inplace=True)
	sched.drop(columns=['VISITOR_PTS', 'HOME_PTS', 'HOME_WIN', 'WINNING_TEAM'], inplace=True)
	merged_df = stats.merge(sched, on=["DATE", "Home Team", "Away Team"])
	merged_df.to_csv("../final_merged_csvs_v2/{}_stats.csv".format(year), index=False)