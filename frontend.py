import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"  # Prevent torch.classes error

import streamlit as st
from sentiment_engine import analyze_input_review as analyze_review

# Page settings
st.set_page_config(page_title="💬 AI Sentiment Analyzer", page_icon="💡", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f8ff;
    }

    .stTextArea textarea {
        font-size: 16px;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #ccc;
    }

    .stButton button {
        background-color: #1f77b4;
        color: white;
        padding: 0.7em 1.5em;
        font-size: 17px;
        border-radius: 10px;
        margin-top: 10px;
    }

    .stButton button:hover {
        background-color: #135d8b;
    }

    .similar-review-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #1f77b4;
    }

    .footer {
        text-align: center;
        font-size: 13px;
        color: #888;
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>💬 AI-Powered Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Classify customer reviews using Google Gemini + FAISS + Transformers</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Input box
review = st.text_area("📝 Write your review:", height=150, placeholder="E.g., I loved the product. It was amazing and super helpful!")

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    if review.strip():
        with st.spinner("Analyzing using AI model..."):
            result = analyze_review(review)
            sentiment = result["predicted_sentiment"].lower()

        st.success("✅ Sentiment Analysis Complete!")
        st.markdown("### 🗣️ **Your Input Review:**")
        st.info(result["input_review"])

        # Sentiment output
        st.markdown("### 🎯 **Predicted Sentiment:**")
        if sentiment == "positive":
            st.success("😊 **Positive** — Looks like a good experience!")
        elif sentiment == "negative":
            st.error("😠 **Negative** — Seems to be a poor experience.")
        elif sentiment == "neutral":
            st.warning("😐 **Neutral** — It’s a balanced opinion.")
        else:
            st.info(f"🔎 **{sentiment.capitalize()}** — Couldn't classify properly.")

        # Similar reviews
        st.markdown("### 🧠 **Similar Reviews (FAISS + Transformers):**")
        for sim in result["similar_reviews"]:
            st.markdown(f"""
                <div class="similar-review-box">
                    <b>{sim['label'].capitalize()} Review:</b><br>
                    {sim['review_text']}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a review first!")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="footer">Made with ❤️ using Streamlit, Gemini API, FAISS, and Transformers</div>', unsafe_allow_html=True)
