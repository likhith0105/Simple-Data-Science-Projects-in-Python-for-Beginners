# ==========================================
# AIRBNB LISTINGS ANALYSIS - FULL PROJECT
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
# Download dataset from Kaggle: AB_NYC_2019.csv
df = pd.read_csv("airbnb_full.csv.csv")

# 3. Basic Exploration
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA INFO ---")
print(df.info())

print("\n--- DESCRIPTION ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 4. Data Cleaning
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Drop unnecessary columns (if exist)
cols_to_drop = ['id', 'name', 'host_name', 'last_review']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Remove outliers (price < 500)
df = df[df['price'] < 500]

print("\n--- CLEANED DATA SHAPE ---")
print(df.shape)

# 5. Data Visualization

# Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.savefig("price_distribution.png")
plt.close()

# Room Type Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='room_type', data=df)
plt.title("Room Type Distribution")
plt.savefig("room_type_distribution.png")
plt.close()

# Price by Neighborhood Group
plt.figure(figsize=(8,5))
sns.boxplot(x='neighbourhood_group', y='price', data=df)
plt.title("Price by Location")
plt.xticks(rotation=30)
plt.savefig("price_by_location.png")
plt.close()

# Availability vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(x='availability_365', y='price', data=df)
plt.title("Availability vs Price")
plt.savefig("availability_vs_price.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# 6. Insights

print("\n--- AVERAGE PRICE BY ROOM TYPE ---")
print(df.groupby('room_type')['price'].mean())

print("\n--- AVERAGE PRICE BY LOCATION ---")
print(df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False))

# Most expensive listing
print("\n--- TOP 5 MOST EXPENSIVE LISTINGS ---")
print(df[['neighbourhood_group','room_type','price']].sort_values(by='price', ascending=False).head())

# 7. Save Cleaned Dataset
df.to_csv("cleaned_airbnb.csv", index=False)

print("\n✅ Project Completed Successfully!")