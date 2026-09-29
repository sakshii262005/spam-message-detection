import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report


# Load UCI SMS Spam Collection
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

print("Total messages:", len(data))

# Input and output
X = data["message"]
y = data["label"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "model.pkl")

print("New model saved as model.pkl")