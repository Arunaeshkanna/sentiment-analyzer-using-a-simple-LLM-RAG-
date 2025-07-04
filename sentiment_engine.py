import os
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Configure APIs
genai.configure(api_key=os.getenv("API_KEY"))
language_model = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Example reviews dataset
reviews_dataset: List[Dict[str, str]] = [
    {"review_text": "I love this product! It's exactly what I needed and works perfectly.", "label": "positive"},
    {"review_text": "Terrible experience. It broke within a week and support was unhelpful.", "label": "negative"},
    {"review_text": "It's okay. Does the job, but nothing special.", "label": "neutral"},
    {"review_text": "Fantastic! The quality exceeded my expectations.", "label": "positive"},
    {"review_text": "Not worth the price. I'm disappointed with the performance.", "label": "negative"},
    {"review_text": "Decent value for the money. Could be better, but not bad.", "label": "neutral"},
]

# Create embeddings for reviews
embedding_matrix = embedder.encode([r["review_text"] for r in reviews_dataset], convert_to_numpy=True, normalize_embeddings=True)
search_index = faiss.IndexFlatL2(embedding_matrix.shape[1])
search_index.add(embedding_matrix)

# Find most similar reviews to an input query
def get_similar_reviews(input_text: str, top_results: int = 3) -> List[Dict[str, str]]:
    query_vec = embedder.encode([input_text], convert_to_numpy=True, normalize_embeddings=True)
    _, nearest_indices = search_index.search(query_vec, top_results)
    return [reviews_dataset[i] for i in nearest_indices[0]]

# Get sentiment prediction using Gemini model
def predict_sentiment(review_text: str) -> str:
    response = language_model.generate_content(f"Classify the sentiment as positive, negative, or neutral:\n\"{review_text}\"")
    return response.text.strip().lower()

# Analyze sentiment and find similar reviews for a given input
def analyze_input_review(text: str) -> Dict:
    similar_reviews = get_similar_reviews(text)
    sentiment = predict_sentiment(text)
    return {
        "input_review": text,
        "similar_reviews": similar_reviews,
        "predicted_sentiment": sentiment
    }

# Evaluate sentiment prediction accuracy on sample dataset
def compute_accuracy() -> Dict:
    correct_count = 0
    total_count = len(reviews_dataset)
    detailed_results = []

    for item in reviews_dataset:
        prediction = predict_sentiment(item["review_text"])
        match = prediction == item["label"]
        detailed_results.append({
            "review_text": item["review_text"],
            "true_label": item["label"],
            "predicted_label": prediction,
            "is_correct": match
        })
        if match:
            correct_count += 1

    accuracy_percentage = round((correct_count / total_count) * 100, 2) if total_count else 0
    return {
        "total_reviews": total_count,
        "correct_predictions": correct_count,
        "accuracy_percent": accuracy_percentage,
        "detailed_results": detailed_results
    }

# Predict sentiment for all reviews (without label check)
def bulk_sentiment_prediction() -> List[Dict[str, str]]:
    predictions = []
    for review in reviews_dataset:
        pred = predict_sentiment(review["review_text"])
        predictions.append({
            "review_text": review["review_text"],
            "predicted_sentiment": pred
        })
    return predictions