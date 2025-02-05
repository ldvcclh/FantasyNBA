from Models.models import Model

def compute_fantasy_score(df):
    """
    WINAMAX fantasy score (miss for now +/- : point marqué/encaissé par l'équipe lorsque le joueur est sur le terrain)
    :param df: DataFrame
    :return:
    Fantasy league points of each player and each game in a new DataFrame column
    """
    # Shots
    three_succeed = df['3P']
    three_missed = df['3PA'] - three_succeed
    two_attemped = df['FGA'] - df['3PA']
    two_succeed = df['FG'] - three_succeed
    two_missed = two_attemped - two_succeed
    ft_succeed = df['FT']                       #free throw
    ft_missed = df['FTA'] - ft_succeed
    # Rebounds
    def_r = df['DRB'] #defensive
    off_r = df['ORB'] #offensive
    # Passes
    assists = df['AST']
    # Steals
    steals = df['STL']
    # Blocks
    blocks = df['BLK']
    # Turnover
    turnover = df['TOV']
    # Fouls
    fouls = df['PF']

    return (two_succeed*2 + two_missed*(-0.5) + three_succeed*3 + three_missed*(-0.5) + ft_succeed + ft_missed*(-0.5)
            + def_r*0.75 + off_r + assists + steals*2 + blocks*2 + turnover*(-0.75) + fouls*(-0.5))


def get_averages(df, num_games,col):
    from tqdm import tqdm
    """
    This function calculates the average for a specified amount of previous games. 
    It also formats the predictive value (next FP scored) for the NN model
    
    :param df: Player DataFrame 
    :param num_games: number of games to calculate the average 
    :param col: column statistics taken into acount for the average computation
    
    :return: New player DataFrame
    """

    # col = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
    #       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'FDP', 'FDS']

    #col = ['MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA',
    #       'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-', 'FP']

    add_col = [f'{num_games}_{i}' for i in col] + ['NFP']

    #player_list = df['Player'].unique()
    #for player in tqdm(player_list):
        # player_df = df[df['Name'] == player].sort_values(by='Date').copy()
    player_df = df.copy()
    # Next fantasy point column
    player_df['NFP'] = player_df['FP'].shift(-1)
    # Add mean of each statistics player
    for c in col:
        player_df[f'{num_games}_{c}'] = player_df[c].rolling(window=num_games, min_periods=num_games).mean()

    return player_df


def fill_players_dict(df, players, col, num_games, model_type,**kwargs):
    """
    Create a dictionary of a list of players with average statistics (col) of n number of games (num_games)
    :param df: Season DataFrame
    :param players: list of players
    :param col: statistics taken into account in the model
    :param num_games: number of games to compute average statistics
    :param model_type: type of model
    :return: dictionary of players with models and samples
    """
    from tqdm import tqdm

    # Init players dictionary
    players_dict = {}

    # Loop over players
    pbar = tqdm(players)
    for player in pbar:
        # Extract player data and get averages statistics
        pbar.set_description(f'Computing average and building model: {player}')
        player_df = df[df['Player'] == player].sort_values(by='Date', ascending=False).copy()
        player_df = player_df[player_df.FP != 0]
        # print("Shape",player_df.shape[0])
        if player_df.shape[0] > num_games + 1:
            player_df = get_averages(player_df, num_games, col)
            player_df = player_df.dropna()

            # Features and output
            x_col = [f'{num_games}_{i}' for i in col]
            X = player_df[[i for i in x_col]]
            y = player_df[['FP']]

            # Init class
            player_model = Model(X, y)

            # Divide data
            player_model.create_train_test_split(test_size=0.3)

            # Construct and train model
            player_model.build_model(model_type,**kwargs)
            player_model.train_model(**kwargs)

            # Evaluate
            metrics = player_model.evaluate_model()

            players_dict[player] = {"model": player_model,"history":player_model.history,
                                    "points": {"x_train": player_model.x_train, "x_test": player_model.x_test,
                                               "y_train": player_model.y_train, "y_test": player_model.y_test,
                                               "y_pred": player_model.y_pred},
                                    "metrics": metrics
                                    }

    return players_dict