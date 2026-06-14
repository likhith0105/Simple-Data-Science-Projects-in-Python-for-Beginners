# Customer Segmentation using K-Means Clustering

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Display first 5 rows
print("First 5 Records:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal K
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Train KMeans Model
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add Cluster Labels
df['Cluster'] = clusters

# Silhouette Score
score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", round(score,3))

# Display Clustered Data
print("\nClustered Customers:")
print(df.head())

# Cluster Centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot Customer Segments
plt.figure(figsize=(8,6))

colors = ['red','blue','green','orange','purple']

for i in range(k):
    plt.scatter(
        df[df['Cluster']==i]['Annual Income (k$)'],
        df[df['Cluster']==i]['Spending Score (1-100)'],
        s=60,
        color=colors[i],
        label=f'Cluster {i+1}'
    )

# Plot Cluster Centers
plt.scatter(
    centers[:,0],
    centers[:,1],
    s=250,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

# Number of customers in each cluster
print("\nCustomers in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())

# Save Output
df.to_csv("Customer_Segmentation_Output.csv", index=False)

print("\nOutput saved as Customer_Segmentation_Output.csv")