
from bref_feature_engineering.rollingavg_seasonlog_teamrecord import teamrecord
from bref_feature_engineering.proportion_ip_available import get_games_played, append_proportion_ip_avail

celtics_17 = teamrecord("Boston Celtics", 2017)
append_proportion_ip_avail(celtics_17, "Boston Celtics")
print(celtics_17)