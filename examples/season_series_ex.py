from bref_feature_engineering.season_series import get_season_series, append_season_series
from bref_feature_engineering.rollingavg_seasonlog_teamrecord import teamrecord

s = teamrecord('Boston Celtics', 2018)
append_season_series(s, "Boston Celtics")
print(s)