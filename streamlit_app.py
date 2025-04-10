import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

st.title("💬 Twitter & Reddit Sentiment Analyzer")
st.subheader("Analyze user sentiment from social media posts")
st.markdown("Upload a post or a CSV file and get quick insights 📊")

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Basic text cleaning (same as your notebook)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict function
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect_text = vectorizer.transform([cleaned])
    pred = model.predict(vect_text)[0]
    return pred

# Streamlit App Layout
st.set_page_config(page_title="Twitter & Reddit Sentiment Analyzer", layout="centered")
st.title("🧠 Twitter & Reddit Sentiment Analyzer")
st.write("A simple NLP app to detect **sentiment** from social media posts using a machine learning model trained on real data.")

# Sidebar
st.sidebar.header("Choose Analysis Mode")
mode = st.sidebar.radio("Mode", ["🔍 Single Text Input", "📂 Upload CSV File"])

# --- SINGLE TEXT MODE ---
if mode == "🔍 Single Text Input":
    text_input = st.text_area("Enter a Tweet or Reddit post:", placeholder="Type your text here...")

    if st.button("Predict Sentiment"):
        if text_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            prediction = predict_sentiment(text_input)
            st.success(f"Predicted Sentiment: **{prediction}**")

# --- BATCH FILE MODE ---
elif mode == "📂 Upload CSV File":
    st.info("Upload a CSV file with a column named `'text'` containing the posts.")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        if 'text' not in df.columns:
            st.error("The uploaded file must contain a 'text' column.")
        else:
            # Clean and predict
            df['cleaned_text'] = df['text'].astype(str).apply(clean_text)
            vect = vectorizer.transform(df['cleaned_text'])
            df['prediction'] = model.predict(vect)

            st.success("Predictions done! Here's a preview:")
            st.dataframe(df[['text', 'prediction']].head(10))

            # Bar Chart
            st.subheader("📊 Sentiment Distribution")
            chart_data = df['prediction'].value_counts()
            st.bar_chart(chart_data)

            # WordCloud
            st.subheader("☁️ WordCloud from Posts")
            all_text = " ".join(df['cleaned_text'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Arpit Singh** | [GitHub](https://github.com/arpitsingh4297)")
