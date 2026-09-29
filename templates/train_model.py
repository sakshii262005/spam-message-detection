import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("spam.csv")

# Input and output
X = data["message"]
y = data["label"]

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
print("Model saved as model.pkl")