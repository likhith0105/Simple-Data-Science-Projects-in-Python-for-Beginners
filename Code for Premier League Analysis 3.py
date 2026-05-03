import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
# Using a public CSV of the 2023/24 season match results
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)

# Keep only essential columns: HomeTeam, AwayTeam, FTHG (Home Goals), FTAG (Away Goals), FTR (Result)
df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

# 2. Calculate Points for each Match
def get_points(result, team_type):
    if result == 'H':
        return 3 if team_type == 'Home' else 0
    elif result == 'A':
        return 0 if team_type == 'Home' else 3
    else:
        return 1

df['HomePoints'] = df['FTR'].apply(lambda x: get_points(x, 'Home'))
df['AwayPoints'] = df['FTR'].apply(lambda x: get_points(x, 'Away'))

# 3. Create the League Table
# Group by Home Team
home_stats = df.groupby('HomeTeam').agg(
    Home_Points=('HomePoints', 'sum'),
    Home_Goals=('FTHG', 'sum')
).reset_index().rename(columns={'HomeTeam': 'Team'})

# Group by Away Team
away_stats = df.groupby('AwayTeam').agg(
    Away_Points=('AwayPoints', 'sum'),
    Away_Goals=('FTAG', 'sum')
).reset_index().rename(columns={'AwayTeam': 'Team'})

# Merge and Calculate Totals
league_table = pd.merge(home_stats, away_stats, on='Team')
league_table['TotalPoints'] = league_table['Home_Points'] + league_table['Away_Points']
league_table = league_table.sort_values(by='TotalPoints', ascending=False)

# 4. Evaluation & Visualization
print("--- Premier League Season Standings (Partial) ---")
print(league_table[['Team', 'TotalPoints']].head(10))

# Visualization: Points Distribution (Home vs Away)
plt.figure(figsize=(12, 6))
top_10 = league_table.head(10)

plt.bar(top_10['Team'], top_10['Home_Points'], label='Home Points', color='skyblue')
plt.bar(top_10['Team'], top_10['Away_Points'], bottom=top_10['Home_Points'], label='Away Points', color='navy')

plt.xticks(rotation=45)
plt.ylabel('Points')
plt.title('Top 10 Teams: Home vs Away Points Distribution')
plt.legend()
plt.tight_layout()
plt.show()