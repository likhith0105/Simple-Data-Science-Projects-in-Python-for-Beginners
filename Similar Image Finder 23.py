import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Extract image features
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    return features.flatten()

# Dataset folder
dataset_path = "dataset"

# Query image
query_image = "query.jpg"

# Extract query features
query_features = extract_features(query_image)

similarities = []

# Compare with all dataset images
for file in os.listdir(dataset_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, file)

        features = extract_features(img_path)

        score = cosine_similarity(
            [query_features],
            [features]
        )[0][0]

        similarities.append((file, score))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

print("\nTop Similar Images:")
for img, score in similarities[:5]:
    print(f"{img} -> Similarity: {score:.4f}")

# Display results
plt.figure(figsize=(15, 5))

query = cv2.cvtColor(cv2.imread(query_image), cv2.COLOR_BGR2RGB)

plt.subplot(1, 6, 1)
plt.imshow(query)
plt.title("Query")
plt.axis("off")

for i, (img_name, score) in enumerate(similarities[:5]):
    img_path = os.path.join(dataset_path, img_name)

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    plt.subplot(1, 6, i + 2)
    plt.imshow(img)
    plt.title(f"{score:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()