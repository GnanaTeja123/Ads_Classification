# Disable TensorFlow for HuggingFace (to prevent DLL errors)
import os
os.environ["USE_TF"] = "0"
streamlit run app.py --server.port $PORT --server.address 0.0.0.0

import streamlit as st
import joblib
import json
import pandas as pd
import evaluate
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load classifier and vectorizer
classifier = joblib.load("model/classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Load evaluation metrics
with open("model/classifier_metrics.json", "r") as f:
    metrics = json.load(f)

# Load HuggingFace summarizer
summarizer = pipeline("summarization", model="t5-small", framework="pt")

# Load ROUGE evaluator
rouge = evaluate.load("rouge")

# --- Streamlit UI ---
st.set_page_config(page_title="AdVision Lite", layout="centered")
st.title("📢 AdVision Lite – Ad Classifier & Summarizer")
st.markdown("Enter the text of an advertisement below:")

# Input box
ad_text = st.text_area("✏️ Enter Ad Text:", height=200)

# Prediction function
def predict_category(texts):
    return classifier.predict(vectorizer.transform(texts))

# Button
if st.button("🔍 Analyze Ad"):
    if ad_text.strip():
        # Predict category
        category = predict_category([ad_text])[0]

        # Generate summary
        summary_output = summarizer(ad_text, max_length=30, min_length=5, do_sample=False)[0]['summary_text']

        # Compute ROUGE score
        rouge_scores = rouge.compute(predictions=[summary_output], references=[ad_text])
        rouge1_f1 = round(rouge_scores["rouge1"], 3)

        # Output results
        st.markdown("### 🧠 Predicted Category")
        st.success(category)

        st.markdown("### ✍️ Generated Summary")
        st.info(summary_output)

        st.markdown("### 📊 Model Performance Metrics")
        st.write(f"**Accuracy:** {metrics['accuracy']}")
        st.write(f"**F1 Score:** {metrics['f1_score']}")
        st.write(f"**ROUGE-1 F1 Score (Summary):** {rouge1_f1}")

        # --- Charts ---

        if "labels" in metrics and "predictions" in metrics:
            # Confusion matrix
            cm = confusion_matrix(metrics["labels"], metrics["predictions"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=sorted(set(metrics["labels"])),
                        yticklabels=sorted(set(metrics["labels"])))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("📉 Confusion Matrix")
            st.pyplot(fig)

            # Distribution chart
            pred_df = pd.DataFrame(metrics["predictions"], columns=["Category"])
            fig2, ax2 = plt.subplots()
            sns.countplot(data=pred_df, x="Category", palette="pastel", ax=ax2)
            plt.title("📊 Predicted Category Distribution")
            plt.xticks(rotation=30)
            st.pyplot(fig2)

    else:
        st.warning("⚠️ Please enter some ad text before analyzing.")

