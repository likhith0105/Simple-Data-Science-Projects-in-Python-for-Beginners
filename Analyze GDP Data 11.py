# ==========================================
# GDP DATA ANALYSIS PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
# Save your CSV file as: gdp_data.csv
df = pd.read_csv("gdp_data.csv")

# Display First 5 Rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Dataset Information
print("\nDATASET INFO:")
print(df.info())

# Check Missing Values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Basic Statistics
print("\nSTATISTICS:")
print(df.describe())

# Rename Columns if Needed
# Example columns:
# Country, Year, GDP

# Plot GDP Trend for All Countries
plt.figure(figsize=(12,6))

countries = df['Country'].unique()

for country in countries:
    country_data = df[df['Country'] == country]
    plt.plot(country_data['Year'],
             country_data['GDP'],
             marker='o',
             label=country)

plt.title("GDP Growth Over Years")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.grid(True)
plt.show()

# Top 10 Highest GDP Countries
latest_year = df['Year'].max()

latest_data = df[df['Year'] == latest_year]

top10 = latest_data.sort_values(by='GDP', ascending=False).head(10)

print("\nTOP 10 GDP COUNTRIES:")
print(top10[['Country', 'GDP']])

# Bar Chart for Top 10 Countries
plt.figure(figsize=(12,6))

plt.bar(top10['Country'], top10['GDP'])

plt.title(f"Top 10 GDP Countries in {latest_year}")
plt.xlabel("Country")
plt.ylabel("GDP")
plt.xticks(rotation=45)

plt.show()

# GDP Growth Percentage
df['GDP_Growth_%'] = df.groupby('Country')['GDP'].pct_change() * 100

print("\nGDP GROWTH PERCENTAGE:")
print(df.head())

# Save Processed File
df.to_csv("processed_gdp_data.csv", index=False)

print("\nProcessed file saved as processed_gdp_data.csv")