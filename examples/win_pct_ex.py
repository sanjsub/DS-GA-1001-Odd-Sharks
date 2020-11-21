from bref_feature_engineering.schedule import get_clean_schedule
from bref_feature_engineering.win_pct import more_cleaning, append_cum_win_pct, append_rolling_win_pct

sched_20 = get_clean_schedule(2020)
more_cleaning(sched_20)
append_cum_win_pct(sched_20)
append_rolling_win_pct(sched_20, 5)
sched_20.to_csv("../schedule_csvs/2020_schedule.csv", index=False)
