# Email Spam Detection using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Sample Dataset
# -----------------------------
data = {
    "email": [
        "Congratulations! You won a free iPhone",
        "Meeting scheduled tomorrow at 10 AM",
        "Claim your free cash reward now",
        "Project report attached",
        "Limited time offer buy now",
        "Your bank account needs verification",
        "Lunch with team today",
        "Win money instantly click here",
        "Assignment submission deadline",
        "Get rich quick scheme"
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Convert Labels
# spam = 1, ham = 0
# -----------------------------
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["email"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_tfidf)

# -----------------------------
# Evaluation
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Custom Prediction Function
# -----------------------------
def predict_email(email_text):
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)[0]

    if prediction == 1:
        return "SPAM"
    else:
        return "NOT SPAM"

# -----------------------------
# User Input
# -----------------------------
while True:
    msg = input("\nEnter email text (or type 'exit'): ")

    if msg.lower() == "exit":
        print("Program terminated.")
        break

    result = predict_email(msg)
    print("Prediction:", result)