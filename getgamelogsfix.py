# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:08:39 2020

@author: sanjs
"""
import pandas as pd
from requests import get
from bs4 import BeautifulSoup
import unicodedata

def get_game_logs_fix(name, start_date, end_date, stat_type = 'BASIC', playoffs=False):
    # Changed to get_player_suffix_fix
    suffix = get_player_suffix_fix(name).replace('/', '%2F').replace('.html', '')
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


def get_player_suffix_fix(name):
    normalized_name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    names = normalized_name.split(' ')[1:]
    for last_name in names:
        initial = last_name[0].lower()
        r = get(f'https://www.basketball-reference.com/players/{initial}')
        if r.status_code==200:
            soup = BeautifulSoup(r.content, 'html.parser')
            for table in soup.find_all('table', attrs={'id': 'players'}):
                suffixes = []              
                for anchor in table.find_all('a'):                    
                    if unicodedata.normalize('NFD', anchor.text).encode('ascii', 'ignore').decode("utf-8").lower() in normalized_name.lower():
                        suffix = anchor.attrs['href']
                        player_r = get(f'https://www.basketball-reference.com{suffix}')
                        if player_r.status_code==200:
                            player_soup = BeautifulSoup(player_r.content, 'html.parser')
                            h1 = player_soup.find('h1', attrs={'itemprop': 'name'})
                            if h1:
                                page_name = h1.find('span').text
                                norm_page_name = unicodedata.normalize('NFD', page_name).encode('ascii', 'ignore').decode("utf-8")
                                if norm_page_name.lower()==normalized_name.lower():
                                    return suffix


