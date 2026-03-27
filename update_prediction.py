import pandas as pd

sub_path = 'data/final_submission.csv'
df = pd.read_csv(sub_path)

# Lehigh ID = 1250, Prairie View ID = 1341
# Team 1 is Lehigh, Team 2 is Prairie View. Prairie View winning means Pred < 0.5.
matchup_id = '2026_1250_1341'

old_pred = df.loc[df['ID'] == matchup_id, 'Pred'].values[0]
df.loc[df['ID'] == matchup_id, 'Pred'] = 0.0  # Force Prairie View to win
new_pred = df.loc[df['ID'] == matchup_id, 'Pred'].values[0]

# Also let's save to a new file so we have a backup, but overwrite final_submission.csv too
df.to_csv(sub_path, index=False)
df.to_csv('final_submission_pvamu.csv', index=False)

print(f"Updated {matchup_id} from {old_pred:.4f} to {new_pred:.4f} (Prairie View wins!)")
