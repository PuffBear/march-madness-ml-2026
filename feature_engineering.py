import pandas as pd
import numpy as np

DATA_DIR = 'data'

def load_data():
    print("Loading datasets...")
    m_reg_detailed = pd.read_csv(f'{DATA_DIR}/MRegularSeasonDetailedResults.csv')
    w_reg_detailed = pd.read_csv(f'{DATA_DIR}/WRegularSeasonDetailedResults.csv')
    m_seeds = pd.read_csv(f'{DATA_DIR}/MNCAATourneySeeds.csv')
    w_seeds = pd.read_csv(f'{DATA_DIR}/WNCAATourneySeeds.csv')
    
    reg_detailed = pd.concat([m_reg_detailed, w_reg_detailed], ignore_index=True)
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
    return reg_detailed, seeds

def calculate_advanced_metrics(df):
    print("Calculating advanced metrics...")
    # Estimate possessions
    w_pos = df['WFGA'] - df['WOR'] + df['WTO'] + 0.475 * df['WFTA']
    l_pos = df['LFGA'] - df['LOR'] + df['LTO'] + 0.475 * df['LFTA']
    df['Possessions'] = (w_pos + l_pos) / 2

    # Efficiencies
    df['WOE'] = (df['WScore'] / df['Possessions']) * 100
    df['WDE'] = (df['LScore'] / df['Possessions']) * 100
    df['LOE'] = (df['LScore'] / df['Possessions']) * 100
    df['LDE'] = (df['WScore'] / df['Possessions']) * 100
    
    # True Shooting Percentage
    df['WTSP'] = df['WScore'] / (2 * (df['WFGA'] + 0.475 * df['WFTA']))
    df['LTSP'] = df['LScore'] / (2 * (df['LFGA'] + 0.475 * df['LFTA']))
    
    # Assist to Turnover Ratio
    df['WAstTORatio'] = df['WAst'] / df['WTO'].replace(0, np.nan)
    df['LAstTORatio'] = df['LAst'] / df['LTO'].replace(0, np.nan)

    # Effective Field Goal Percentage
    df['WeFGP'] = (df['WFGM'] + 0.5 * df['WFGM3']) / df['WFGA']
    df['LeFGP'] = (df['LFGM'] + 0.5 * df['LFGM3']) / df['LFGA']
    
    # --- NEW FEATURES ---
    
    # Offensive Rebound Rate: OR / (OR + Opp_DR)
    df['WORP'] = df['WOR'] / (df['WOR'] + df['LDR']).replace(0, np.nan)
    df['LORP'] = df['LOR'] / (df['LOR'] + df['WDR']).replace(0, np.nan)
    
    # Defensive Rebound Rate: DR / (DR + Opp_OR)
    df['WDRP'] = df['WDR'] / (df['WDR'] + df['LOR']).replace(0, np.nan)
    df['LDRP'] = df['LDR'] / (df['LDR'] + df['WOR']).replace(0, np.nan)
    
    # Steal Rate: Stl / Possessions
    df['WStlRate'] = df['WStl'] / df['Possessions']
    df['LStlRate'] = df['LStl'] / df['Possessions']
    
    # Block Rate: Blk / Opp_FGA
    df['WBlkRate'] = df['WBlk'] / df['LFGA'].replace(0, np.nan)
    df['LBlkRate'] = df['LBlk'] / df['WFGA'].replace(0, np.nan)
    
    # Free Throw Rate: FTA / FGA
    df['WFTR'] = df['WFTA'] / df['WFGA'].replace(0, np.nan)
    df['LFTR'] = df['LFTA'] / df['LFGA'].replace(0, np.nan)
    
    # 3-Point Attempt Rate: FGA3 / FGA
    df['W3PAr'] = df['WFGA3'] / df['WFGA'].replace(0, np.nan)
    df['L3PAr'] = df['LFGA3'] / df['LFGA'].replace(0, np.nan)
    
    return df

def aggregate_season_stats(df):
    print("Aggregating team season statistics...")
    
    w_cols = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore', 'WOE', 'WDE', 'WTSP', 
              'WAstTORatio', 'WeFGP', 'Possessions', 'WORP', 'WDRP', 'WStlRate', 'WBlkRate', 'WFTR', 'W3PAr',
              'LTeamID']
    l_cols = ['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore', 'LOE', 'LDE', 'LTSP', 
              'LAstTORatio', 'LeFGP', 'Possessions', 'LORP', 'LDRP', 'LStlRate', 'LBlkRate', 'LFTR', 'L3PAr',
              'WTeamID']
    
    common_names = ['Season', 'DayNum', 'TeamID', 'PointsScored', 'PointsAllowed', 'OE', 'DE', 'TSP', 
                    'AstTORatio', 'eFGP', 'Possessions', 'ORP', 'DRP', 'StlRate', 'BlkRate', 'FTR', 'ThreePAr',
                    'OpponentID']
    
    winner_stats = df[w_cols].copy()
    winner_stats.columns = common_names
    winner_stats['Wins'] = 1

    loser_stats = df[l_cols].copy()
    loser_stats.columns = common_names
    loser_stats['Wins'] = 0

    all_games = pd.concat([winner_stats, loser_stats], ignore_index=True)
    
    # --- Strength of Schedule ---
    # First compute raw WinPct per team per season
    raw_win_pct = all_games.groupby(['Season', 'TeamID'])['Wins'].mean().reset_index()
    raw_win_pct.rename(columns={'Wins': 'OppWinPct'}, inplace=True)
    
    # Merge opponent win pct onto each game
    all_games = all_games.merge(raw_win_pct, left_on=['Season', 'OpponentID'], 
                                right_on=['Season', 'TeamID'], how='left', suffixes=('', '_opp'))
    all_games.drop('TeamID_opp', axis=1, inplace=True)
    
    # --- Recent Form (last 14 days of season, DayNum > 118) ---
    recent_games = all_games[all_games['DayNum'] > 118].copy()
    recent_form = recent_games.groupby(['Season', 'TeamID']).agg(
        RecentWinPct=('Wins', 'mean'),
        RecentOE=('OE', 'mean')
    ).reset_index()

    # --- Full season aggregation ---
    team_season_stats = all_games.groupby(['Season', 'TeamID']).agg(
        AvgPointsScored=('PointsScored', 'mean'),
        AvgPointsAllowed=('PointsAllowed', 'mean'),
        AvgOE=('OE', 'mean'),
        AvgDE=('DE', 'mean'),
        AvgTSP=('TSP', 'mean'),
        AvgAstTORatio=('AstTORatio', 'mean'),
        AvgeFGP=('eFGP', 'mean'),
        AvgPossessions=('Possessions', 'mean'),
        AvgORP=('ORP', 'mean'),
        AvgDRP=('DRP', 'mean'),
        AvgStlRate=('StlRate', 'mean'),
        AvgBlkRate=('BlkRate', 'mean'),
        AvgFTR=('FTR', 'mean'),
        Avg3PAr=('ThreePAr', 'mean'),
        SOS=('OppWinPct', 'mean'),
        WinPct=('Wins', 'mean'),
        GamesPlayed=('Wins', 'count')
    ).reset_index()

    team_season_stats['NetRating'] = team_season_stats['AvgOE'] - team_season_stats['AvgDE']
    
    # Merge recent form
    team_season_stats = team_season_stats.merge(recent_form, on=['Season', 'TeamID'], how='left')
    # Fill teams with no recent games with their season averages
    team_season_stats['RecentWinPct'] = team_season_stats['RecentWinPct'].fillna(team_season_stats['WinPct'])
    team_season_stats['RecentOE'] = team_season_stats['RecentOE'].fillna(team_season_stats['AvgOE'])
    
    return team_season_stats

def add_seed_info(team_stats, seeds):
    print("Merging seed info...")
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    team_stats = team_stats.merge(seeds[['Season', 'TeamID', 'SeedNum']], on=['Season', 'TeamID'], how='left')
    team_stats['SeedNum'] = team_stats['SeedNum'].fillna(20)
    return team_stats

def add_elo_info(team_stats):
    print("Merging Elo ratings...")
    elo_df = pd.read_csv(f'{DATA_DIR}/elo_ratings.csv')
    team_stats = team_stats.merge(elo_df, on=['Season', 'TeamID'], how='left')
    team_stats['Elo'] = team_stats['Elo'].fillna(1500)
    return team_stats

def main():
    reg_detailed, seeds = load_data()
    reg_detailed = calculate_advanced_metrics(reg_detailed)
    team_season_stats = aggregate_season_stats(reg_detailed)
    team_season_stats = add_seed_info(team_season_stats, seeds)
    team_season_stats = add_elo_info(team_season_stats)
    
    output_path = f'{DATA_DIR}/team_season_features.csv'
    team_season_stats.to_csv(output_path, index=False)
    print(f"Features saved to {output_path} (Shape: {team_season_stats.shape})")

if __name__ == '__main__':
    main()
