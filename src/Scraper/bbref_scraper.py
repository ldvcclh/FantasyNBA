import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import socket
import os
import pickle
import re

class BbrefScraper:
    def __init__(self, season_start_links, scrape_type):
        self.scrape_type = scrape_type
        self.season_start_links = season_start_links
        self.base_url = 'https://www.basketball-reference.com'
        self.api_key = None
        if len(self.season_start_links)==1:
            match = re.search(r'\d+', self.season_start_links[0])
            if match:
                self.year = int(match.group())

        computer_name = socket.gethostname()

        self.data_path = '/Users/julialegrand/PycharmProjects/FantasyNBA/data'


        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            os.mkdir(f'{self.data_path}/pickles')
            os.mkdir(f'{self.data_path}/bbref-files')

        self.team_dictionary = {
            'Atlanta Hawks': 'Atl', 'Boston Celtics': 'Bos', 'Brooklyn Nets': 'Bkn', 'Charlotte Hornets': 'Cha',
            'Chicago Bulls': 'Chi', 'Cleveland Cavaliers': 'Cle', 'Dallas Mavericks': 'Dal', 'Denver Nuggets': 'Den',
            'Detroit Pistons': 'Det', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'Hou', 'Indiana Pacers': 'Ind',
            'Los Angeles Lakers': 'LAL', 'Los Angeles Clippers': 'LAC', 'Memphis Grizzlies': 'Mem', 'Miami Heat': 'Mia',
            'Milwaukee Bucks': 'Mil', 'Minnesota Timberwolves': 'Min', 'New Orleans Pelicans': 'Nor', 'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'Orl', 'Philadelphia 76ers': 'Phi', 'Phoenix Suns': 'Pho',
            'Portland Trail Blazers': 'Por', 'Sacramento Kings': 'Sac', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'Tor',
            'Utah Jazz': 'Uta', 'Washington Wizards': 'Was'
        }

    def scrape_stats(self):
        print(f'Scraping\n','~' * 50, '\n', '\n'.join(self.season_start_links))
        #get the url for each month in the season
        print("START get the url for each month in the season")
        self.get_months()
        print("get the url for each month in the season DONE")
        #print(self.month_url_dict)

        #get the game links for each game within each month
        print("START get the url for each month in the season")
        self.get_game_links()
        print("get the game links for each game within each month DONE")
        #print(self.full_game_urls)

        #scrape each table in each game and save to dataframe
        self.get_game_stats()

    def get_game_stats(self):
        self.full_game_urls = pickle.load(open(f'{self.data_path}/pickles/GameLinks_{self.year}.p', 'rb'))

        for season, game_links in self.full_game_urls.items():
            pieces = []
            pbar = tqdm(game_links, desc = f'Scraping Games: {season}')
            # total = 0
            for link in pbar:
                print('Game Link',link)
                scraper_api_url = f"https://api.scraperapi.com/?api_key={self.api_key}&url={link}"
                response = requests.get(scraper_api_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    time.sleep(1)
                else:
                    print(f"Failed to load page: {response.status_code}")
                    return
                #soup = BeautifulSoup(response.text, 'html.parser')
                table_df = self.get_table_info(soup, link)
                pieces.append(table_df)
                # total += 1
                # if total == 50:
                #     break
            full_season_df = pd.concat(pieces)
            full_season_df.to_csv(f'{self.data_path}/bbref-files/{season}.csv', index = False)

    def get_table_info(self, soup, link):
        columns = ['Player', 'Date', 'Team', 'Against', 'Home', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
                   'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-']
        # column_2 = ['TS%', 'eFG%',
        #            'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM']
        header = soup.findAll('h1')[0].text.split(',')
        #print("header",header)
        date = ''.join(header[-2:]).strip()
        teams = header[0].split('at')
        #print('TEAMS',teams)
        # away_team = teams[0].strip()
        # home_team = teams[1][:teams[1].find('Box Score')].strip()
        away_team = teams[0].strip()
        home_team = teams[-1][:teams[-1].find('Box Score')].strip()
        #print("away",away_team)
        #print("home", home_team)
        if home_team == '':
            home_team = teams[-2][:teams[-2].find('Box Score')].strip()
            #print("home v2", home_team)
        tables = soup.findAll('tbody')
        #get unqiue players
        player_dict = {}
        away_idx = [0]
        home_idx = [8]
        for idx, table in enumerate(tables):
            #print("idx",idx)
            #print("table",table)
            #print('~'*50)
            if idx not in away_idx + home_idx:
                continue
            if idx in away_idx:
                team = away_team
                opp = home_team
                home = 0
            else:
                team = home_team
                opp = away_team
                home = 1
            for row in table:
                try:
                    name = row.findAll('th')[0].text
                    if name == 'Reserves':
                        continue
                    player_dict[name] = [name, date, team, opp, home]
                    #print("Player", player_dict[name])
                    #print("ID",id)
                    #print("row",row)
                except AttributeError:
                    continue
        for idx, table in enumerate(tables):
            if idx not in away_idx + home_idx:
                continue
            for row in table:
                try:
                    name = row.findAll('th')[0].text
                    cols = row.findAll('td')
                    if len(cols) == 0:
                        continue
                    cols = [i.text.strip() for i in cols]
                    for c in cols:
                        player_dict[name].append(c)
                    #print("Player v2", player_dict[name])
                except AttributeError:
                    continue
        game_df_pieces = []
        for name, l in player_dict.items():
            #print("Name, l",name,l)
            row_dict = {col:value for col, value in zip(columns, l)}
            row_df = pd.DataFrame(row_dict, index = [0])
            #print("row df",row_df)
            game_df_pieces.append(row_df)
        game_df = pd.concat(game_df_pieces)
        game_df['GameLink'] = [link for i in range(len(game_df))]
        # game_df.to_csv('Tester.csv', index = False)
        # game_df.to_csv('Tester.csv', index = False)
        return game_df

    def get_game_links(self):
        self.full_game_urls = {}
        pbar = tqdm((enumerate(self.month_url_dict.items())), total = len(self.month_url_dict))
        for idx, (season, month_url_list) in pbar:
            pbar.set_description(f'({idx+1}-{len(self.month_url_dict)}) | Getting Game URLs: {season}')
            season_game_urls = []
            for month_url in month_url_list:
                response = requests.get(month_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    time.sleep(1)
                else:
                    print(f"Failed to load page: {response.status_code}")
                    return
                #soup = BeautifulSoup(response.text, 'html.parser')
                time.sleep(1)
                table = soup.findAll('tbody')
                # print(table)
                game_links = table[0].findAll('a', href = True)
                for idx, link in enumerate(game_links):
                    if (idx + 1) % 4 != 0:
                        continue
                    if link.text.strip() != 'Box Score':
                        continue
                    full_url = f'{self.base_url}{link["href"]}'
                    season_game_urls.append(full_url)

            self.full_game_urls[season] = season_game_urls
        pbar.close()
        pickle.dump(self.full_game_urls, open(f'{self.data_path}/pickles/GameLinks_{self.year}.p', 'wb'))
        del self.month_url_dict

    def get_months(self):
        if self.scrape_type == 0:
            months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
        elif self.scrape_type == 1:
            months = ['november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
        elif self.scrape_type == 2:
            months = ['december', 'january', 'february', 'march', 'april', 'may', 'june']
        elif self.scrape_type == 3:
            months = ['october-2019', 'november', 'december', 'january', 'february', 'march', 'july', 'august', 'september', 'october-2020']
        elif self.scrape_type == 4:
            months = ['october', 'november','december', 'january', 'february', 'march']
        self.month_url_dict = {}

        for season_start_link in self.season_start_links:
            print("Link",season_start_link)
            payload = {'api_key': self.api_key,
                       'url': season_start_link}
            response = requests.get('https://api.scraperapi.com/', params=payload)
            #response = requests.get(season_start_link)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
            else:
                print(f"Failed to load page: {response.status_code}")
                return
            #soup = BeautifulSoup(response.text, 'html.parser')
            print("H1",soup.find('h1'))
            season = soup.find('h1').text.strip().split(' ')[0]#.strip()
            #print(soup)
            print("Season",season)
            if os.path.exists(f'{self.data_path}/bbref-files/{season}.csv'):
                continue
            self.year = f'20{season[season.find("-")+1:]}'
            season_month_list = []
            for month in months:
                base_url = f'https://www.basketball-reference.com/leagues/NBA_{self.year}_games-{month}.html'
                season_month_list.append(base_url)
            self.month_url_dict[season] = season_month_list
        # print(self.month_url_dict[season])

if __name__ == '__main__':

    season_list_3 = [
        'https://www.basketball-reference.com/leagues/NBA_2020_games.html'
    ]
    season_list_2 = [
        'https://www.basketball-reference.com/leagues/NBA_2012_games.html',
    ]
    season_list_1 = [
        'https://www.basketball-reference.com/leagues/NBA_2005_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2006_games.html',
    ]       #doesnt work with current method

    season_list_0 = [
        'https://www.basketball-reference.com/leagues/NBA_2001_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2002_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2003_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2004_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2007_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2008_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2009_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2010_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2011_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2013_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2014_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2015_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2016_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2017_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2018_games.html',
        'https://www.basketball-reference.com/leagues/NBA_2019_games.html',
    ]
    season_dict = {0: season_list_0, 1: season_list_1, 2: season_list_2, 3: season_list_3}
    total = len(season_list_0) + len(season_list_1) + len(season_list_2) + len(season_list_3)
    # print(f'Total Games: {total}')
    # for scrape_type, season_list in season_dict.items():
        # nba_stat_scraper = BbrefScraper(season_list, scrape_type = scrape_type)
        # nba_stat_scraper.scrape_stats()

    season_list = [
       'https://www.basketball-reference.com/leagues/NBA_2022_games.html'
    ]

    nba_stat_scraper = BbrefScraper(season_list, scrape_type = 4)
    nba_stat_scraper.scrape_stats()