#%%
import pandas as pd
'''
seasons_df = []
years_range = sorted(range(2008, 2020), reverse=True)
for i in years_range:
    filename = "../" + str(i) + "_season.csv"
    seasons_df.append(pd.read_csv(filename))


all_seasons_df = pd.concat(seasons_df)

outfile = "all_seasons.csv"
all_seasons_df.to_csv(outfile, index=False)

all_seasons_df
'''
# %%
from os import listdir
from os.path import isfile, join
import datetime 

other_odds_files = [f for f in listdir("Historical Odds/") if isfile(join("Historical Odds/", f))]
other_odds_df = []
years_range = sorted(range(2008, 2009)) ## intentionally skipping 2019 for now

for filename, year in zip(other_odds_files[:-1], years_range):
    scores = []
    filename = "Historical Odds/" + filename
    other_odds_season_df = pd.read_excel(filename)
    
    bookies_file = "../" + str(year) + "_season.csv"
    bookies_df = pd.read_csv(bookies_file, parse_dates=['Game Date'])
    
    day_of_month = other_odds_season_df['Date'] % 100
    month = other_odds_season_df['Date'] // 100
    year = (month < 10) + year
    
    dates_df_true = pd.to_datetime({'year': year, 'month': month, 'day': day_of_month})
    dates_df_plus1 = dates_df_true + pd.Timedelta(days=1)
    dates_df_less1 = dates_df_true + pd.Timedelta(days=-1)
    
    ## Create unique column for team key

    other_odds_season_df['Team Num'] = other_odds_season_df['Team'].apply(lambda x: other_odds_team_to_num[x])

    for index, row in bookies_df['Game Date'].items():
        relevant_games = other_odds_season_df[(dates_df_plus1 >= row) & (dates_df_less1 <= row)]
        visitor_games = relevant_games[relevant_games['VH']=="V"]
        home_games = relevant_games[relevant_games['VH']=="H"]
        
        away_bookie_num = bookie_team_to_num[bookies_df.loc[index]['Away Team']]
        home_bookie = bookie_team_to_num[bookies_df.loc[index]['Home Team']]

        ## Need to figure out how exactly to tackle duplicates in visitors and home
        ## Need to also deal with "N" values in the visitor home

        print('HERES ROW:\n')
        print(bookies_df.loc[index])
        print()
        print(visitor_games)
        print(home_games)
        # print(relevant_games['Team'])
        #print(index, row)
        if index > 1:
            break

    #other_odds_df.append(other_odds_df)
    ## Iterate rows, do pandas splicing with booleans, then compare visitor, home to make
    ## sure it is the right game, etc

    # still might have to edit this

#%%
'''
len(other_odds_df)
other_odds_files
# other_odds_df
print(other_odds_df[0].loc[0]['Date'])
#other_odds_df[0]['Date'] > 1000
day_of_month = other_odds_df[0]['Date'] % 100
month = other_odds_df[0]['Date'] // 100
year = (month < 10) + 2008
test = other_odds_df[0]


dates_df_true = pd.to_datetime({'year': year, 'month': month, 'day': day_of_month})
dates_df_plus1 = dates_df_true + pd.Timedelta(days=1)
dates_df_less1 = dates_df_true + pd.Timedelta(days=-1)
# random_date = datetime.datetime.strptime("30-01-2009", "%d-%m-%Y")
'''
#%%
'''
all_seasons_other_df = pd.concat(other_odds_df, sort=False).drop(columns=['Unnamed: 13','Unnamed: 14','Unnamed: 15'])

all_seasons_other_df

outfile = "all_seasons_other.csv"
all_seasons_other_df.to_csv(outfile, index=False)
'''
# %%

bookie_team_to_num = {'Atlanta Hawks':		1,
'Boston Celtics':		2,
'Brooklyn Nets':		3,
'Charlotte Hornets':		4,
'Chicago Bulls':		5,
'Cleveland Cavaliers':		6,
'Dallas Mavericks':		7,
'Denver Nuggets':		8,
'Detroit Pistons':		9,
'Golden State Warriors':		10,
'Houston Rockets':		11,
'Indiana Pacers':		12,
'Los Angeles Clippers':		13,
'Los Angeles Lakers':		14,
'Memphis Grizzlies':		15,
'Miami Heat':		16,
'Milwaukee Bucks':		17,
'Minnesota Timberwolves':		18,
'New Orleans Pelicans':		19,
'New York Knicks':		20,
'Oklahoma City Thunder':		21,
'Orlando Magic':		22,
'Philadelphia 76ers':		23,
'Phoenix Suns':		24,
'Portland Trail Blazers':		25,
'Sacramento Kings':		26,
'San Antonio Spurs':		27,
'Toronto Raptors':		28,
'Utah Jazz':		29,
'Washington Wizards':		30}

other_odds_team_to_num = {
'Atlanta':	1,
'Boston':	2,
'Brooklyn':	3,
'Charlotte':	4,
'Chicago':	5,
'Cleveland':	6,
'Dallas':	7,
'Denver':	8,
'Detroit':	9,
'GoldenState':	10,
'Houston':	11,
'Indiana':	12,
'LA Clippers':	13,
'LAClippers':	13,
'LALakers':	14,
'Memphis':	15,
'Miami':	16,
'Milwaukee':	17,
'Minnesota':	18,
'NewJersey':	3,
'NewOrleans':	19,
'NewYork':	20,
'Oklahoma City':	21,
'OklahomaCity':	21,
'Orlando':	22,
'Philadelphia':	23,
'Phoenix':	24,
'Portland':	25,
'Sacramento':	26,
'SanAntonio':	27,
'Toronto':	28,
'Utah':	29,
'Washington':	30,
}

# %%
