import pandas as pd
import numpy as np

# Load processed data
df = pd.read_csv('processed_deliveries.csv')

# --- Current Score ---
# innings_score already represents the cumulative runs scored by the chasing team

# --- Runs Left ---
df['runs_left'] = (df['target_score'] - df['innings_score']).clip(lower=0)

# --- Balls Left ---
# over is 0-indexed (0–19), ball is 1-indexed (1–6)
# Total balls in an innings = 120, balls bowled so far = over * 6 + ball
df['balls_left'] = (126 - (df['over'] * 6 + df['ball'])).clip(lower=0)

# --- Wickets Left ---
df['wickets_left'] = 10 - df['innings_wickets']

# --- Current Run Rate (CRR) ---
balls_bowled = 120 - df['balls_left']
overs_bowled = balls_bowled / 6
df['crr'] = (df['innings_score'] / overs_bowled).replace([np.inf, -np.inf], 0).fillna(0)

# --- Required Run Rate (RRR) ---
overs_remaining = df['balls_left'] / 6
df['rrr'] = (df['runs_left'] / overs_remaining).replace([np.inf, -np.inf], 0).fillna(0)

# --- Final Dataframe ---
# Keep match_id for leakage-free group splits during training. It is not used as a model feature.
final_df = df[['match_id', 'batting_team', 'bowling_team', 'venue',
               'runs_left', 'balls_left', 'wickets_left',
               'target_score', 'crr', 'rrr', 'result']].copy()

# --- Shuffle & Save ---
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv('ipl_model_ready.csv', index=False)

print(f"Total rows: {len(final_df)}")
print(f"\nHead:\n{final_df.head(10)}")
print(f"\nSaved to ipl_model_ready.csv")
