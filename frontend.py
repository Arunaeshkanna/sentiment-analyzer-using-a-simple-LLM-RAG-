import streamlit as st
from sentiment_engine import analyze_input_review as analyze_review

# Page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# Custom CSS for advanced styling
st.markdown("""
    <style>
    /* Page background and layout */
    .main {
        background-color: #f7fbff;
        padding: 2rem;
    }

    /* Title styling */
    h1 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
        font-weight: bold;
    }

    /* Text area */
    .stTextArea textarea {
        font-size: 16px;
        background-color: #ffffff;
        color: #333333;
        border-radius: 10px;
        border: 1px solid #dcdcdc;
        padding: 1rem;
        line-height: 1.5;
    }

    /* Button style */
    .stButton button {
        background-color: #0066cc;
        color: white;
        font-size: 17px;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.8em;
        margin-top: 1em;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #004a99;
    }

    /* Info & Result Box */
    .stAlert {
        border-radius: 10px;
        font-size: 16px;
    }

    /* Divider */
    hr {
        border-top: 2px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>💬 Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Analyze the sentiment of your review with a single click 🔍</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Input
review = st.text_area("📝 Write your review below:", height=150, placeholder="Eg: I loved the product! Very helpful and high quality.")

# Button
analyze = st.button("🔍 Analyze Sentiment")

# Result
if analyze:
    if review.strip():
        with st.spinner("🧠 Analyzing your review..."):
            result = analyze_review(review)
            st.success("✅ Sentiment Analysis Complete!")

            st.markdown("---")
            st.markdown("### 🗣️ **Your Input Review:**")
            st.info(result["input_review"])

            st.markdown("### 🎯 **Predicted Sentiment:**")
            sentiment = result["predicted_sentiment"].lower()

            if sentiment == "positive":
                st.success("😊 **Positive** — Great! The review reflects a happy experience.")
            elif sentiment == "negative":
                st.error("😠 **Negative** — Uh oh! This review seems to reflect a bad experience.")
            elif sentiment == "neutral":
                st.warning("😐 **Neutral** — It’s a balanced or indifferent opinion.")
            else:
                st.info(f"🔎 **{sentiment.capitalize()}** — Couldn’t classify clearly.")

    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px; color: grey;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
