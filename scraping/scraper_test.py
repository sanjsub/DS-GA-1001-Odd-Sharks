#%%
from bs4 import BeautifulSoup
import requests
import datetime
from datetime import date

'''
This is current scrap code to get all the games for 2019 - 2020. We then get all URLs for 
all seasons and then repeat the scraping, write to CSV files and store
'''


url = 'https://www.betexplorer.com/basketball/usa/nba/results/?stage=bPyjUVy1&month=all'
url_2018 = 'https://www.betexplorer.com/basketball/usa/nba-2018-2019/results/?stage=8Y7SorQp&month=all'
url_2016 = 'https://www.betexplorer.com/basketball/usa/nba-2016-2017/results/?stage=zPq9t1nb&month=all'

unique_identifier = 'table-main h-mb15 js-tablebanner-t js-tablebanner-ntb'

headers = {
    'cookie': 'js_cookie=1; my_cookie_id=101112686; my_cookie_hash=94b719f17654f45fad8f47626803f40a; upcoming=3-200; my_timezone=^%^2B1; widget_timeStamp=1576892147; page_cached=1; widget_pageViewCount=12',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'referer': 'https://www.betexplorer.com',
    'authority': 'www.betexplorer.com',
    'x-requested-with': 'XMLHttpRequest',
}





# %%


page = requests.get(url_2016)
soup = BeautifulSoup(page.text, "html.parser")
table = soup.find("table", attrs={"class":"table-main h-mb15 js-tablebanner-t js-tablebanner-ntb"})
rows = table.find_all("tr")
# %%
## what each row looks like
count = 0
for row in rows:
    cols = row.find_all("td")
    versus = cols[0].a.find_all("span")
    home, away = versus[0].text, versus[1].text 
    date = cols[4].text.split(".")

    ### ADD this datetime line when we figure out how to automate each date each year
    ## should be very easy fix depending on how we implement our functions
    ## each individual game_page probably has the date / time
    # dt = datetime(int(date[2]), int(date[1]), int(date[0])).timestamp()
    if not date[2]:
        date[2] = "YEAR IS MISSING" ## datetime.now().year    <---- The old value
    if not cols[0].a.has_attr("href"):
        continue
    link = cols[0].a.attrs["href"]
    game_id = link.split("/")[-2]
    game_page = requests.get(f"https://www.betexplorer.com/match-odds/{game_id}/1/ha/", headers=headers)
    soup = BeautifulSoup(game_page.json()["odds"], "html.parser")
    table = soup.find("table", attrs={"class":"table-main h-mb15 sortable"})
    if not table:
        continue
    bookmakers = table.find_all("tr")[1:-1]
    if not bookmakers: # there were no bookmakers for this game
        continue
    
    ## Now it looks like we are exploring each bookmaker
    ## The github code has 4 bookmakers that the guy seemed interested in but 
    # we'll just take all for now until we know if we'd like to select out (bad?) bookies
    for b in bookmakers:
        if not b.has_attr("data-bid"):
            continue
        bid = b.attrs['data-bid']
        ## if bid not in bookies:
        ##      continue
        cols = b.find_all("td")
        
        if not cols[3].has_attr("data-opening-odd"):
            cols[3].attrs["data-opening-odd"] = "NO OPENING ODDS" ## in caleb's code was set to 1.0
        if not cols[4].has_attr("data-odd"):
            cols[4].attrs["data-odd"] = "NO DATA ODDS" ## in caleb's code was set to 1.0
        
        home_open, home_close = cols[3].attrs["data-opening-odd"], cols[3].attrs["data-odd"]
        away_open, away_close = cols[4].attrs["data-opening-odd"], cols[4].attrs["data-odd"]

        print(home_close, away_close)


    print("____________")
    count += 1    ######## Keeping counting restricting to test code 
    if count > 1:
        break
# %%
