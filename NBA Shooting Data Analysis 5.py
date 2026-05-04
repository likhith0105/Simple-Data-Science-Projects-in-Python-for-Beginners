# ==============================
# NBA SHOOTING DATA ANALYSIS
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
# Example file: nba_shooting.csv
df = pd.read_csv("nba_shooting.csv")

# 3. Display Data
print(df.head())
print(df.info())

# 4. Data Cleaning
df.dropna(inplace=True)

# 5. Basic Columns Expected
# Player, Team, FG%, 3P%, FT%, Points

# 6. Top 10 Players by Field Goal %
top_fg = df.sort_values(by='FG%', ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x='FG%', y='Player', data=top_fg)
plt.title("Top 10 Players by Field Goal %")
plt.show()

# 7. Top 10 Players by 3-Point %
top_3p = df.sort_values(by='3P%', ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x='3P%', y='Player', data=top_3p)
plt.title("Top 10 Players by 3-Point %")
plt.show()

# 8. Shooting Efficiency Comparison
plt.figure(figsize=(6,5))
sns.scatterplot(x='FG%', y='Points', data=df)
plt.title("FG% vs Points")
plt.xlabel("Field Goal %")
plt.ylabel("Points")
plt.show()

# 9. Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# 10. Team-wise Average Shooting
team_avg = df.groupby('Team')[['FG%', '3P%', 'FT%']].mean()

print("\nTeam-wise Shooting Averages:\n", team_avg)

team_avg.plot(kind='bar', figsize=(10,6))
plt.title("Team Shooting Comparison")
plt.ylabel("Percentage")
plt.show()

# 11. Insights
print("\nInsights:")
print("- Players with high FG% often score more points.")
print("- 3P% varies significantly across players.")
print("- Team performance depends on balanced shooting stats.")