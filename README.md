# 🧠 Twitter & Reddit Sentiment Analysis Using NLP

![cover](https://github.com/arpitsingh4297/twitter-reddit-nlp-sentiment-analysis/blob/main/assets/banner.png)

## 📌 Project Overview

This project aims to analyze sentiments from tweets and Reddit posts using Natural Language Processing (NLP). By classifying public opinions into **positive**, **neutral**, and **negative**, it helps understand the collective voice on social platforms. It combines NLP techniques, machine learning models, and explainability methods to build an insightful and production-ready sentiment analysis pipeline.

## 🚀 Problem Statement

Social media platforms like Twitter and Reddit contain millions of user-generated posts that reflect public sentiment on various topics. However, analyzing this unstructured text data at scale is a challenge.

**Goal:** Build an end-to-end pipeline that can ingest raw text from Twitter & Reddit, clean and preprocess it, apply sentiment classification using NLP and ML models, and generate actionable insights with explainability and dashboards.

---

## 📚 Key Learnings

- Hands-on experience with **NLP pipelines**: cleaning, tokenization, vectorization (TF-IDF).
- Built and compared **multiple ML classifiers**: Logistic Regression, SVM, Random Forest, XGBoost.
- Integrated **SHAP Explainability** to understand model decisions.
- Visualized insights using **Seaborn**, **Matplotlib**, and saved EDA plots.
- Understood the importance of **balanced datasets**, performance metrics, and error analysis.

---

## 🧰 Technologies Used

| Category                 | Tools & Libraries                                   |
|--------------------------|-----------------------------------------------------|
| Programming              | Python                                               |
| Data Manipulation        | pandas, numpy                                        |
| Visualization            | seaborn, matplotlib                                  |
| NLP                      | nltk, re, string                                     |
| Feature Engineering      | TF-IDF, CountVectorizer                              |
| ML Algorithms            | Logistic Regression, SVM, Random Forest, XGBoost     |
| Model Evaluation         | accuracy, precision, recall, F1-score, confusion matrix |
| Explainability           | SHAP                                                 |
| Dashboard                | Streamlit                                            |
| Deployment Ready         | Streamlit App with live interactive classification  |

---

## 📊 Project Workflow

```
📁 Twitter_Reddit_Sentiment_Analysis
│
├── 📂 data/
│   ├── twitter_data.csv
│   ├── reddit_data.csv
│
├── 📂 notebooks/
│   ├── Twitter_Reddit_NLP_EDA_Modeling.ipynb
│
├── 📂 models/
│   ├── final_model.pkl
│
├── 📂 plots/
│   ├── eda_label_distribution.png
│   ├── confusion_matrix.png
│
├── 📂 app/
│   ├── streamlit_app.py
│
├── 📂 reports/
│   ├── Business_Report_Twitter_Reddit_Sentiment.docx
│   ├── Sentiment_Analysis_Presentation.pptx
│
├── README.md
└── requirements.txt
```

---

## 📈 Model Performance

| Model              | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.85     | 0.86      | 0.84   | 0.85     |
| SVM                | 0.87     | 0.88      | 0.86   | 0.87     |
| Random Forest      | 0.82     | 0.83      | 0.81   | 0.82     |
| XGBoost            | 0.89 ✅  | 0.90      | 0.88   | 0.89     |

✅ Final model: **XGBoost Classifier** – Tuned and deployed.

---

## 🔍 SHAP Explainability

- SHAP plots were used to interpret how each word/feature contributes to the sentiment prediction.
- Increased model transparency and trust for non-technical stakeholders.

---

## 💻 Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/arpitsingh4297/twitter-reddit-nlp-sentiment-analysis.git

# Step 2: Install dependencies
cd twitter-reddit-nlp-sentiment-analysis
pip install -r requirements.txt

# Step 3: Run Streamlit app
streamlit run app/streamlit_app.py
```

---

## 📄 Reports & Presentation

- ✔️ **Business Report** – Detailed document highlighting business problem, methodology, insights & recommendations.
- ✔️ **Presentation Slides** – Visual summary for stakeholders & interviews.
- ✔️ **Interactive Dashboard** – Deployed Streamlit app for real-time sentiment classification.

---

## 📌 GitHub Repo

🔗 [**GitHub Repository**](https://github.com/arpitsingh4297/twitter-reddit-nlp-sentiment-analysis)

---

## 🙌 Acknowledgements

This project is part of my **Data Science & Business Analytics Portfolio**. Special thanks to the open-source community and contributors of `nltk`, `scikit-learn`, `xgboost`, and `SHAP`.

---

## 📬 Connect with Me

- **LinkedIn**: [Arpit Singh](https://www.linkedin.com/in/arpitsingh4297)
- **GitHub**: [@arpitsingh4297](https://github.com/arpitsingh4297)
- **Portfolio Projects**: [Click here](https://github.com/arpitsingh4297?tab=repositories)
