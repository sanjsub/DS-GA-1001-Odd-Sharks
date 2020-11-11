import pandas as pd
from basketball_reference_scraper.utils import get_player_suffix
from requests import get
from bs4 import BeautifulSoup

# in order to run -- from gamelogsfix import get_game_logs_fix
def get_game_logs_fix(name, start_date, end_date, stat_type = 'BASIC', playoffs=False):
    suffix = get_player_suffix(name).replace('/', '%2F').replace('.html', '')
    start_date_str = start_date
    end_date_str = end_date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    years = list(range(start_date.year, end_date.year+2))
    if stat_type == 'BASIC':
        if playoffs:
            selector = 'div_pgl_basic_playoffs'
        else:
            selector = 'div_pgl_basic'
    elif stat_type == 'ADVANCED':
        if playoffs:
            selector = 'div_pgl_advanced_playoffs'
        else:
            selector = 'div_pgl_advanced'
    final_df = None
    for year in years:
        if stat_type == 'BASIC':
            r = get(f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url={suffix}%2Fgamelog%2F{year}&div={selector}')
        elif stat_type == 'ADVANCED':
            r = get(f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url={suffix}%2Fgamelog-advanced%2F{year}&div={selector}')
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



def get_schedule_fix(season, playoffs=False):
    months = ['October', 'November', 'December', 'January', 'February', 'March', 'April', 'May', 'June']
    if season==2020:
        months = ['October', 'November', 'December', 'January', 'February', 'March', 'July', 'August', 'September', 'October']
    df = pd.DataFrame()
    for month in months:
        r = get(f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month.lower()}.html')
        if r.status_code==200:
            soup = BeautifulSoup(r.content, 'html.parser')
            table = soup.find('table', attrs={'id': 'schedule'})
            month_df = pd.read_html(str(table))[0]
            df = df.append(month_df)
    df = df.reset_index()
    cols_to_remove = [i for i in df.columns if 'Unnamed' in i]
    cols_to_remove += [i for i in df.columns if 'Notes' in i]
    cols_to_remove += [i for i in df.columns if 'Start' in i]
    cols_to_remove += [i for i in df.columns if 'Attend' in i]
    cols_to_remove += ['index']
    df = df.drop(cols_to_remove, axis=1)
    df.columns = ['DATE', 'VISITOR', 'VISITOR_PTS', 'HOME', 'HOME_PTS']
    if season==2020:
        df = df[df['DATE']!='Playoffs']
        df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x))
        df = df.sort_values(by='DATE')
        df = df.reset_index().drop('index', axis=1)
        playoff_loc = df[df['DATE']==pd.to_datetime('2020-08-17')].head(n=1)
        if len(playoff_loc.index)>0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        if playoffs:
            df = df[playoff_index:]
        else:
            df = df[:playoff_index]
    else:
        playoff_loc = df[df['DATE']=='Playoffs']
        if len(playoff_loc.index)>0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        if playoffs:
            df = df[playoff_index+1:]
        else:
            df = df[:playoff_index]
        df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x))
    return df
