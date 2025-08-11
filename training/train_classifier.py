import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your CSV (must contain 'text' and 'category')
df = pd.read_csv(r"C:\Users\gnana\OneDrive\Desktop\advision_new\data\sample_ad_texts.csv")
X = df["text"]
y = df["category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Predictions
y_pred = clf.predict(X_test_vec)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass

# Save model and vectorizer
joblib.dump(clf, "../app/model/classifier.pkl")
joblib.dump(vectorizer, "../app/model/vectorizer.pkl")

# Save metrics with label info
metrics = {
    "accuracy": round(accuracy, 3),
    "f1_score": round(f1, 3),
    "labels": list(y_test),
    "predictions": list(y_pred)
}
with open("../app/model/classifier_metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ Model training complete. Metrics saved.")
