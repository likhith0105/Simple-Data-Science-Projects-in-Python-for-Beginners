import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("premier_league.csv")

# Display basic information
print(df.head())
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Match Result Distribution
sns.countplot(x='FTR', data=df)
plt.title('Match Result Distribution')
plt.xlabel('Result (H=Home Win, D=Draw, A=Away Win)')
plt.show()

# Goals scored by home teams
plt.figure(figsize=(8,5))
sns.histplot(df['FTHG'], bins=10, kde=True)
plt.title('Distribution of Home Goals')
plt.show()

# Goals scored by away teams
plt.figure(figsize=(8,5))
sns.histplot(df['FTAG'], bins=10, kde=True)
plt.title('Distribution of Away Goals')
plt.show()

# Total goals per match
df['TotalGoals'] = df['FTHG'] + df['FTAG']

plt.figure(figsize=(8,5))
sns.histplot(df['TotalGoals'], bins=15, kde=True)
plt.title('Total Goals per Match')
plt.show()

# Home team performance
home_points = {}

for _, row in df.iterrows():
    team = row['HomeTeam']

    if team not in home_points:
        home_points[team] = 0

    if row['FTR'] == 'H':
        home_points[team] += 3
    elif row['FTR'] == 'D':
        home_points[team] += 1

home_df = pd.DataFrame(
    home_points.items(),
    columns=['Team', 'HomePoints']
).sort_values(by='HomePoints', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=home_df.head(10),
            x='HomePoints',
            y='Team')
plt.title('Top 10 Teams by Home Points')
plt.show()

# Correlation Heatmap
numeric_cols = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(),
            annot=True,
            cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()