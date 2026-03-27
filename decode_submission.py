import pandas as pd
import os

# Define paths
data_dir = '/Users/Agriya/Desktop/marchmania/data'
sub_path = os.path.join(data_dir, 'final_submission.csv')
m_teams_path = os.path.join(data_dir, 'MTeams.csv')
w_teams_path = os.path.join(data_dir, 'WTeams.csv')
out_path = '/Users/Agriya/Desktop/marchmania/decoded_submission.csv'

# Read files
sub_df = pd.read_csv(sub_path)
m_teams = pd.read_csv(m_teams_path)[['TeamID', 'TeamName']]
w_teams = pd.read_csv(w_teams_path)[['TeamID', 'TeamName']]

# Combine M and W teams since their IDs are mutually exclusive (1000s vs 3000s)
all_teams = pd.concat([m_teams, w_teams]).set_index('TeamID')['TeamName'].to_dict()

# Extract Year, Team1, Team2
sub_df[['Year', 'Team1ID', 'Team2ID']] = sub_df['ID'].str.split('_', expand=True).astype(int)

# Map names
sub_df['Team1Name'] = sub_df['Team1ID'].map(all_teams)
sub_df['Team2Name'] = sub_df['Team2ID'].map(all_teams)

# Reorder columns and format output
output_df = sub_df[['Year', 'Team1Name', 'Team2Name', 'Pred']]
output_df.to_csv(out_path, index=False)

print(f"Decoded {len(output_df)} rows and saved to {out_path}.")
print("\nFirst few rows:")
print(output_df.head())
