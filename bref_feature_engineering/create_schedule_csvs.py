from bref_feature_engineering.schedule import get_clean_schedule
from bref_feature_engineering.win_pct import more_cleaning, append_cum_win_pct, append_rolling_win_pct

for year in range(2009, 2020):
	sched = get_clean_schedule(year)
	more_cleaning(sched)
	append_cum_win_pct(sched)
	append_rolling_win_pct(sched, 5)
	sched.to_csv("../schedule_csvs/{}_schedule.csv".format(year), index=False)