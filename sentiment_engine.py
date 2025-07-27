import os
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

# Init models
language_model = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Sample dataset
reviews_dataset: List[Dict[str, str]] = [
    {"review_text": "I love this product! It's exactly what I needed and works perfectly.", "label": "positive"},
    {"review_text": "Terrible experience. It broke within a week and support was unhelpful.", "label": "negative"},
    {"review_text": "It's okay. Does the job, but nothing special.", "label": "neutral"},
    {"review_text": "Fantastic! The quality exceeded my expectations.", "label": "positive"},
    {"review_text": "Not worth the price. I'm disappointed with the performance.", "label": "negative"},
    {"review_text": "Decent value for the money. Could be better, but not bad.", "label": "neutral"},
]

# Embedding + FAISS
embedding_matrix = embedder.encode([r["review_text"] for r in reviews_dataset], convert_to_numpy=True, normalize_embeddings=True)
search_index = faiss.IndexFlatL2(embedding_matrix.shape[1])
search_index.add(embedding_matrix)

def get_similar_reviews(input_text: str, top_results: int = 3) -> List[Dict[str, str]]:
    query_vec = embedder.encode([input_text], convert_to_numpy=True, normalize_embeddings=True)
    _, nearest_indices = search_index.search(query_vec, top_results)
    return [reviews_dataset[i] for i in nearest_indices[0]]

def predict_sentiment(review_text: str) -> str:
    prompt = f"Classify the sentiment as positive, negative, or neutral:\n\"{review_text}\""
    response = language_model.generate_content(prompt)
    return response.text.strip().lower()

def analyze_input_review(text: str) -> Dict:
    similar_reviews = get_similar_reviews(text)
    sentiment = predict_sentiment(text)
    return {
        "input_review": text,
        "similar_reviews": similar_reviews,
        "predicted_sentiment": sentiment
    }
