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

    for index, row in bookies_df['Game Date'].items():
        print(other_odds_season_df[(dates_df_plus1 >= row) & (dates_df_less1 <= row)]) 
        
        print(row)
        print(filename, bookies_file)
        print()
        print()
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

#%%
'''
all_seasons_other_df = pd.concat(other_odds_df, sort=False).drop(columns=['Unnamed: 13','Unnamed: 14','Unnamed: 15'])

all_seasons_other_df

outfile = "all_seasons_other.csv"
all_seasons_other_df.to_csv(outfile, index=False)
'''
# %%
