import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Extract color histogram features
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Dataset folder
dataset_folder = "images"

image_paths = []
features = []

for file in os.listdir(dataset_folder):
    path = os.path.join(dataset_folder, file)
    image_paths.append(path)
    features.append(extract_features(path))

features = np.array(features)

# Query image
query_image = "query.jpg"
query_feature = extract_features(query_image)

# Similarity calculation
similarities = cosine_similarity(
    [query_feature],
    features
)[0]

# Top 5 similar images
top_indices = np.argsort(similarities)[::-1][:5]

print("Most Similar Images:")
for idx in top_indices:
    print(image_paths[idx], similarities[idx])

# Display results
plt.figure(figsize=(12, 4))

for i, idx in enumerate(top_indices):
    img = cv2.imread(image_paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(f"{similarities[idx]:.2f}")
    plt.axis("off")

plt.show()