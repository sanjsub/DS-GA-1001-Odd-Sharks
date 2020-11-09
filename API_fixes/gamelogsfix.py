import pandas as pd
from basketball_reference_scraper.utils import get_player_suffix
from requests import get
from bs4 import BeautifulSoup

# in order to run -- from gamelogsfix import get_game_logs_fix

def get_game_logs_fix(name, start_date, end_date, playoffs=False):
    suffix = get_player_suffix(name).replace('/', '%2F').replace('.html', '')
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
                df.rename(columns = {'Date': 'DATE', 'Age': 'AGE', 'Tm': 'TEAM', 'Unnamed: 5': 'HOME/AWAY', 'Opp': 'OPPONENT',
                        'Unnamed: 7': 'RESULT', 'GmSc': 'GAME_SCORE'}, inplace=True)
                df['HOME/AWAY'] = df['HOME/AWAY'].apply(lambda x: 'AWAY' if x=='@' else 'HOME')
                df = df[df['Rk']!='Rk']
                #df = df.drop(['Rk', 'G'], axis=1)
                df = df.loc[(df['DATE'] >= start_date_str) & (df['DATE'] <= end_date_str)]
                active_df = pd.DataFrame(columns = list(df.columns))
                for index, row in df.iterrows():
                    if row['GS']=='Inactive' or row['GS']=='Did Not Play' or row['GS']=='Not With Team' or row['GS']=='Did Not Dress':
                        continue
                    active_df = active_df.append(row)
                if final_df is None:
                    final_df = pd.DataFrame(columns=list(active_df.columns))
                final_df = final_df.append(active_df)
    final_df.set_index(['Rk'], inplace = True)
    return final_df

gamelogfix = get_game_logs_fix('Moritz Wagner', '2018-10-18', '2019-04-09', playoffs=False)
