import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("spam.csv")

print("Total messages:", len(data))
print("\nDataset columns:", data.columns.tolist())
print("\nClass distribution:")
print(data["label"].value_counts())


# -----------------------------
# 2. Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["message"] = data["message"].apply(clean_text)


# -----------------------------
# 3. Input and Output
# -----------------------------
X = data["message"]
y = data["label"]


# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5. ML Pipeline
# -----------------------------
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        MultinomialNB(alpha=0.1)
    )
])


# -----------------------------
# 6. Train Model
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# 7. Test Model
# -----------------------------
predictions = model.predict(X_test)


# -----------------------------
# 8. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# -----------------------------
# 9. Save Model
# -----------------------------
joblib.dump(model, "model.pkl")

print("\nNew model saved successfully as model.pkl")