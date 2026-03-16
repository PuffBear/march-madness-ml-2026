"""
Custom Elo Rating System for NCAA Basketball.

Processes every game chronologically. Each team starts at 1500 Elo.
After each game, the winner gains points and the loser loses points,
scaled by how surprising the result was.

At the end of each season, ratings regress 25% toward the mean (1500)
to account for roster turnover / graduation.
"""

import pandas as pd
import numpy as np

DATA_DIR = 'data'

# Elo parameters
INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 100
SEASON_REVERSION = 0.25  # 25% mean reversion between seasons

def expected_score(elo_a, elo_b):
    """Calculate expected win probability for team A."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

def update_elo(winner_elo, loser_elo, k=K_FACTOR):
    """Update Elo ratings after a game. Returns (new_winner_elo, new_loser_elo)."""
    exp_w = expected_score(winner_elo, loser_elo)
    exp_l = 1.0 - exp_w
    
    new_winner = winner_elo + k * (1.0 - exp_w)
    new_loser = loser_elo + k * (0.0 - exp_l)
    
    return new_winner, new_loser

def build_elo_ratings():
    print("Building Elo ratings from game-by-game data...")
    
    # Load ALL compact results (regular season + tournament) for both M and W
    m_reg = pd.read_csv(f'{DATA_DIR}/MRegularSeasonCompactResults.csv')
    w_reg = pd.read_csv(f'{DATA_DIR}/WRegularSeasonCompactResults.csv')
    m_tourney = pd.read_csv(f'{DATA_DIR}/MNCAATourneyCompactResults.csv')
    w_tourney = pd.read_csv(f'{DATA_DIR}/WNCAATourneyCompactResults.csv')
    
    # Combine and sort chronologically
    all_games = pd.concat([m_reg, w_reg, m_tourney, w_tourney], ignore_index=True)
    all_games = all_games.sort_values(['Season', 'DayNum']).reset_index(drop=True)
    
    # Track Elo ratings
    elo_ratings = {}  # TeamID -> current Elo
    
    # Store end-of-regular-season Elo snapshots for feature engineering
    # We snapshot at DayNum 132 (just before tournament starts)
    season_elo_snapshots = []
    
    current_season = None
    
    for _, game in all_games.iterrows():
        season = game['Season']
        day = game['DayNum']
        winner = game['WTeamID']
        loser = game['LTeamID']
        wloc = game.get('WLoc', 'N')
        
        # Season change: apply mean reversion
        if season != current_season:
            if current_season is not None:
                # Snapshot all ratings at end of previous season's regular season
                # (We already captured them via the DayNum check below)
                
                # Apply mean reversion for new season
                for team_id in elo_ratings:
                    elo_ratings[team_id] = (
                        elo_ratings[team_id] * (1 - SEASON_REVERSION) + 
                        INITIAL_ELO * SEASON_REVERSION
                    )
            current_season = season
        
        # Initialize teams if first time seeing them
        if winner not in elo_ratings:
            elo_ratings[winner] = INITIAL_ELO
        if loser not in elo_ratings:
            elo_ratings[loser] = INITIAL_ELO
        
        # Get current ratings
        w_elo = elo_ratings[winner]
        l_elo = elo_ratings[loser]
        
        # Adjust for home court advantage (only in regular season)
        if wloc == 'H':
            w_elo_adj = w_elo + HOME_ADVANTAGE
            l_elo_adj = l_elo
        elif wloc == 'A':
            w_elo_adj = w_elo
            l_elo_adj = l_elo + HOME_ADVANTAGE
        else:
            w_elo_adj = w_elo
            l_elo_adj = l_elo
        
        # Update Elo (using adjusted for calculation, but store unadjusted)
        exp_w = expected_score(w_elo_adj, l_elo_adj)
        
        elo_ratings[winner] = w_elo + K_FACTOR * (1.0 - exp_w)
        elo_ratings[loser] = l_elo + K_FACTOR * (0.0 - (1.0 - exp_w))
    
    # Now do a second pass to capture end-of-regular-season snapshots properly
    print("Capturing pre-tournament Elo snapshots...")
    elo_ratings_pass2 = {}
    current_season = None
    snapshots_taken = set()
    
    for _, game in all_games.iterrows():
        season = game['Season']
        day = game['DayNum']
        winner = game['WTeamID']
        loser = game['LTeamID']
        wloc = game.get('WLoc', 'N')
        
        if season != current_season:
            # Before changing season, snapshot everything for previous season
            if current_season is not None and current_season not in snapshots_taken:
                for team_id, elo in elo_ratings_pass2.items():
                    season_elo_snapshots.append({
                        'Season': current_season,
                        'TeamID': team_id,
                        'Elo': elo
                    })
                snapshots_taken.add(current_season)
                
                # Mean reversion
                for team_id in elo_ratings_pass2:
                    elo_ratings_pass2[team_id] = (
                        elo_ratings_pass2[team_id] * (1 - SEASON_REVERSION) + 
                        INITIAL_ELO * SEASON_REVERSION
                    )
            current_season = season
        
        if winner not in elo_ratings_pass2:
            elo_ratings_pass2[winner] = INITIAL_ELO
        if loser not in elo_ratings_pass2:
            elo_ratings_pass2[loser] = INITIAL_ELO
        
        # Only process regular season games (DayNum < 136) for the snapshot
        if day < 136:
            w_elo = elo_ratings_pass2[winner]
            l_elo = elo_ratings_pass2[loser]
            
            if wloc == 'H':
                w_elo_adj = w_elo + HOME_ADVANTAGE
                l_elo_adj = l_elo
            elif wloc == 'A':
                w_elo_adj = w_elo
                l_elo_adj = l_elo + HOME_ADVANTAGE
            else:
                w_elo_adj = w_elo
                l_elo_adj = l_elo
            
            exp_w = expected_score(w_elo_adj, l_elo_adj)
            elo_ratings_pass2[winner] = w_elo + K_FACTOR * (1.0 - exp_w)
            elo_ratings_pass2[loser] = l_elo + K_FACTOR * (0.0 - (1.0 - exp_w))
        
    # Snapshot final season
    if current_season not in snapshots_taken:
        for team_id, elo in elo_ratings_pass2.items():
            season_elo_snapshots.append({
                'Season': current_season,
                'TeamID': team_id,
                'Elo': elo
            })
    
    elo_df = pd.DataFrame(season_elo_snapshots)
    
    output_path = f'{DATA_DIR}/elo_ratings.csv'
    elo_df.to_csv(output_path, index=False)
    print(f"Elo ratings saved to {output_path} (Shape: {elo_df.shape})")
    
    return elo_df

if __name__ == '__main__':
    build_elo_ratings()
