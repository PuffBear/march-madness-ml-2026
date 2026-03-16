import pandas as pd
import numpy as np

DATA_DIR = 'data'

def load_data():
    print("Loading required datasets for building training data...")
    m_tourney_results = pd.read_csv(f'{DATA_DIR}/MNCAATourneyCompactResults.csv')
    w_tourney_results = pd.read_csv(f'{DATA_DIR}/WNCAATourneyCompactResults.csv')
    tourney_results = pd.concat([m_tourney_results, w_tourney_results], ignore_index=True)
    features = pd.read_csv(f'{DATA_DIR}/team_season_features.csv')
    return tourney_results, features

def create_symmetric_matchups(tourney_results):
    print("Creating symmetric matchup dataset...")
    # We want to double the dataset: one row where Team1 won, another where Team1 lost
    
    # Win perspective
    df_win = tourney_results[['Season', 'WTeamID', 'LTeamID']].copy()
    df_win.rename(columns={'WTeamID': 'Team1', 'LTeamID': 'Team2'}, inplace=True)
    df_win['Target'] = 1
    
    # Loss perspective (swap Team1 and Team2)
    df_loss = tourney_results[['Season', 'LTeamID', 'WTeamID']].copy()
    df_loss.rename(columns={'LTeamID': 'Team1', 'WTeamID': 'Team2'}, inplace=True)
    df_loss['Target'] = 0
    
    # Combine
    matchups = pd.concat([df_win, df_loss], ignore_index=True)
    
    # Shuffle to mix wins and losses
    matchups = matchups.sample(frac=1, random_state=42).reset_index(drop=True)
    return matchups

def merge_features(matchups, features):
    print("Merging features and computing differences...")
    
    # Merge Team1 features
    df = matchups.merge(features, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
    df.drop('TeamID', axis=1, inplace=True)
    
    # Rename T1 features
    rename_dict_t1 = {col: f"{col}_T1" for col in features.columns if col not in ['Season', 'TeamID']}
    df.rename(columns=rename_dict_t1, inplace=True)
    
    # Merge Team2 features
    df = df.merge(features, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')
    df.drop('TeamID', axis=1, inplace=True)
    
    # Rename T2 features
    rename_dict_t2 = {col: f"{col}_T2" for col in features.columns if col not in ['Season', 'TeamID']}
    df.rename(columns=rename_dict_t2, inplace=True)
    
    # Compute feature differences (Team1 - Team2)
    feature_cols = [col for col in features.columns if col not in ['Season', 'TeamID']]
    for col in feature_cols:
        df[f"{col}_Diff"] = df[f"{col}_T1"] - df[f"{col}_T2"]
        
    # Drop absolute T1 and T2 columns to enforce symmetric learning purely on differences
    cols_to_drop = [f"{col}_T1" for col in feature_cols] + [f"{col}_T2" for col in feature_cols]
    df.drop(columns=cols_to_drop, inplace=True)
    
    return df

def main():
    tourney_results, features = load_data()
    matchups = create_symmetric_matchups(tourney_results)
    training_data = merge_features(matchups, features)
    
    # Detailed results (and therefore advanced stats) only exist from 2003 onwards
    # We must drop NaN rows for seasons prior to 2003
    initial_shape = training_data.shape
    training_data.dropna(inplace=True)
    
    output_path = f'{DATA_DIR}/training_dataset.csv'
    training_data.to_csv(output_path, index=False)
    print(f"Dropped {initial_shape[0] - training_data.shape[0]} old matches lacking detailed stats.")
    print(f"Training dataset saved to {output_path} with final shape {training_data.shape}")

if __name__ == '__main__':
    main()
