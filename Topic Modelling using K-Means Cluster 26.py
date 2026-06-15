from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

documents = [
    "Machine learning algorithms analyze data",
    "Deep learning is part of artificial intelligence",
    "Stock markets involve financial investments",
    "Investors buy and sell stocks",
    "Hospitals provide healthcare services"
]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply K-Means
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Print cluster assignments
for i, label in enumerate(kmeans.labels_):
    print(f"Document {i+1}: Cluster {label}")