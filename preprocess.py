import pandas as pd

# Load data
df = pd.read_csv('archive/ipl_data.csv')

# --- Assign innings numbers ---
# For each match, the first batting_team encountered is innings 1, the second is innings 2.
# Any further innings (Super Overs) are dropped.
def assign_innings(group):
    teams_seen = []
    innings = []
    for team in group['batting_team']:
        if team not in teams_seen:
            teams_seen.append(team)
        idx = teams_seen.index(team) + 1
        innings.append(idx)
    group = group.copy()
    group['innings'] = innings
    return group

df = df.groupby('match_id', group_keys=False).apply(assign_innings)

# Keep only standard 1st and 2nd innings (drop Super Overs etc.)
df = df[df['innings'].isin([1, 2])]

# --- Clean Team Names ---
team_name_map = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
}

for col in ['batting_team', 'bowling_team', 'match_winner']:
    df[col] = df[col].replace(team_name_map)

# --- Filter Venues ---
# Remove rows with missing venue data
df = df.dropna(subset=['venue'])
df = df[df['venue'].str.strip() != '']

# --- Calculate Total Runs per 1st Innings ---
first_innings = df[df['innings'] == 1]
first_innings_total = first_innings.groupby('match_id')['runs_total'].sum().reset_index()
first_innings_total.columns = ['match_id', 'first_innings_runs']

# --- Build 2nd Innings dataframe with target_score ---
second_innings = df[df['innings'] == 2].copy()
second_innings = second_innings.merge(first_innings_total, on='match_id', how='left')
second_innings['target_score'] = second_innings['first_innings_runs'] + 1

# --- Label the Result ---
second_innings['result'] = (second_innings['batting_team'] == second_innings['match_winner']).astype(int)

# --- Save ---
second_innings.to_csv('processed_deliveries.csv', index=False)

print(f"Processed {second_innings['match_id'].nunique()} matches, {len(second_innings)} deliveries.")
print(f"Saved to processed_deliveries.csv")
print(f"\nSample columns: {list(second_innings.columns)}")
print(f"\nTeams:\n{second_innings['batting_team'].unique()}")
