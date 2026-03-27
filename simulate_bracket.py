import pandas as pd
import sys

# Load team IDs
teams_df = pd.read_csv('data/MTeams.csv')
name_to_id = {row['TeamName']: row['TeamID'] for _, row in teams_df.iterrows()}

aliases = {
    'UConn': 'Connecticut',
    'Prairie View': 'Prairie View',
    'Saint Mary\'s': "St Mary's CA",
    'McNeese': 'McNeese St',
    'Long Island': 'LIU Brooklyn',
    'Kennesaw St.': 'Kennesaw',
    'Saint Louis': 'St Louis',
    'Queens (N.C.)': 'Queens NC',
    'NC ST': 'NC State',
    'TEXAS': 'Texas',
    'MIA OH': 'Miami OH',
    'HOW': 'Howard',
    'LEHIGH': 'Lehigh',
    'PVAMU': 'Prairie View',
    'UMBC': 'UMBC',
    'SMU': 'SMU',
    'Miami (FL)': 'Miami FL',
    'North Dakota St.': 'N Dakota St',
}

def get_team_id(name):
    original_name = name
    name = aliases.get(name, name)
    
    if name in name_to_id: return name, name_to_id[name]
    name_no_dot = name.replace('.', '')
    if name_no_dot in name_to_id: return name_no_dot, name_to_id[name_no_dot]
    if name.endswith(' St.'):
        name_st = name[:-4] + ' St'
        if name_st in name_to_id: return name_st, name_to_id[name_st]
            
    print(f"ERROR: Could not find team {original_name} (mapped to {name})")
    sys.exit(1)

# Load predictions directly from final submission
preds_df = pd.read_csv('data/final_submission.csv')
pred_map = {}
for _, row in preds_df.iterrows():
    # Only map Men's tournaments starting with 2026_1
    if not str(row['ID']).startswith('2026_1'): continue
    pred_map[row['ID']] = row['Pred']

def play_game(teamA, teamB):
    nameA, idA = get_team_id(teamA)
    nameB, idB = get_team_id(teamB)
    
    if idA < idB:
        matchp_id = f"2026_{idA}_{idB}"
        p = pred_map.get(matchp_id)
        if p is None: return teamA
        if p > 0.5: return teamA
        else: return teamB
    else:
        matchp_id = f"2026_{idB}_{idA}"
        p = pred_map.get(matchp_id)
        if p is None: return teamB
        if p > 0.5: return teamB
        else: return teamA

def get_prob(teamA, teamB):
    nameA, idA = get_team_id(teamA)
    nameB, idB = get_team_id(teamB)
    if idA < idB:
        p = pred_map.get(f"2026_{idA}_{idB}", 0.5)
        return p if p > 0.5 else 1 - p
    else:
        p = pred_map.get(f"2026_{idB}_{idA}", 0.5)
        return p if p > 0.5 else 1 - p

def print_round(teams, round_name):
    print(f"\n--- {round_name} ---")
    next_round = []
    for i in range(0, len(teams), 2):
        w = play_game(teams[i], teams[i+1])
        prob = get_prob(teams[i], teams[i+1]) * 100
        print(f"{teams[i]:<16} vs {teams[i+1]:<16} -> {w:<16} ({prob:.1f}%)")
        next_round.append(w)
    return next_round

print("=== FIRST FOUR ===")
s_16 = play_game('LEHIGH', 'PVAMU')
print(f"SOUTH 16: LEHIGH vs PVAMU -> {s_16}")
w_11 = play_game('NC ST', 'TEXAS')
print(f"WEST 11: NC ST vs TEXAS -> {w_11}")
mw_16 = play_game('HOW', 'UMBC')
print(f"MIDWEST 16: HOW vs UMBC -> {mw_16}")
mw_11 = play_game('SMU', 'MIA OH')
print(f"MIDWEST 11: SMU vs MIA OH -> {mw_11}")

east_teams = ['Duke', 'Siena', 'Ohio St.', 'TCU', 'St. John\'s', 'Northern Iowa', 'Kansas', 'Cal Baptist', 'Louisville', 'South Florida', 'Michigan St.', 'North Dakota St.', 'UCLA', 'UCF', 'UConn', 'Furman']
south_teams = ['Florida', s_16, 'Clemson', 'Iowa', 'Vanderbilt', 'McNeese', 'Nebraska', 'Troy', 'North Carolina', 'VCU', 'Illinois', 'Penn', 'Saint Mary\'s', 'Texas A&M', 'Houston', 'Idaho']
west_teams = ['Arizona', 'Long Island', 'Villanova', 'Utah St.', 'Wisconsin', 'High Point', 'Arkansas', 'Hawaii', 'BYU', w_11, 'Gonzaga', 'Kennesaw St.', 'Miami (FL)', 'Missouri', 'Purdue', 'Queens (N.C.)']
midwest_teams = ['Michigan', mw_16, 'Georgia', 'Saint Louis', 'Texas Tech', 'Akron', 'Alabama', 'Hofstra', 'Tennessee', mw_11, 'Virginia', 'Wright St.', 'Kentucky', 'Santa Clara', 'Iowa St.', 'Tennessee St.']

print("\n================== REGIONALS ==================")
print("\n=== EAST REGION ===")
e_r32 = print_round(east_teams, "Round of 64")
e_s16 = print_round(e_r32, "Round of 32")
e_e8 = print_round(e_s16, "Sweet 16")
e_f4 = print_round(e_e8, "Elite 8")
east_champ = e_f4[0]

print("\n=== SOUTH REGION ===")
s_r32 = print_round(south_teams, "Round of 64")
s_s16 = print_round(s_r32, "Round of 32")
s_e8 = print_round(s_s16, "Sweet 16")
s_f4 = print_round(s_e8, "Elite 8")
south_champ = s_f4[0]

print("\n=== WEST REGION ===")
w_r32 = print_round(west_teams, "Round of 64")
w_s16 = print_round(w_r32, "Round of 32")
w_e8 = print_round(w_s16, "Sweet 16")
w_f4 = print_round(w_e8, "Elite 8")
west_champ = w_f4[0]

print("\n=== MIDWEST REGION ===")
mw_r32 = print_round(midwest_teams, "Round of 64")
mw_s16 = print_round(mw_r32, "Round of 32")
mw_e8 = print_round(mw_s16, "Sweet 16")
mw_f4 = print_round(mw_e8, "Elite 8")
midwest_champ = mw_f4[0]

print("\n================== FINAL FOUR ==================")
print(f"East Champ: {east_champ}")
print(f"South Champ: {south_champ}")
print(f"West Champ: {west_champ}")
print(f"Midwest Champ: {midwest_champ}")

print("\n--- National Semifinals ---")
sf1_winner = play_game(east_champ, south_champ)
print(f"Semifinal 1: {east_champ} vs {south_champ} -> {sf1_winner}")
sf2_winner = play_game(west_champ, midwest_champ)
print(f"Semifinal 2: {west_champ} vs {midwest_champ} -> {sf2_winner}")

print("\n--- National Championship ---")
champion = play_game(sf1_winner, sf2_winner)
print(f"{sf1_winner} vs {sf2_winner} -> {champion} WINS IT ALL!")
