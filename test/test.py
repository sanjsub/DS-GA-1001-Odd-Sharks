#%%
import numpy as np
import pandas as pd
from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats
from basketball_reference_scraper.players import get_stats, get_game_logs
from basketball_reference_scraper.box_scores import get_box_scores
from basketball_reference_scraper.injury_report import get_injury_report
from requests import get
from bs4 import BeautifulSoup



g = get_game_logs("Lebron James", '2014-10-30', '2015-04-13', playoffs=False)
# %%
g
# %%
def get_game_logs2(name, start_date, end_date, playoffs=False):
    # suffix = get_player_suffix(name).replace('/', '%2F').replace('.html', '')
    start_date_str = start_date
    end_date_str = end_date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    years = list(range(start_date.year, end_date.year+2))
    if playoffs:
        selector = 'div_pgl_basic_playoffs'
    else:
        selector = 'div_pgl_basic'
    final_df = None
    for year in years:
        r = get(f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url={suffix}%2Fgamelog%2F{year}&div={selector}')
        if r.status_code==200:
            soup = BeautifulSoup(r.content, 'html.parser')
            table = soup.find('table')
            if table:
                df = pd.read_html(str(table))[0]
    return df
# %%
