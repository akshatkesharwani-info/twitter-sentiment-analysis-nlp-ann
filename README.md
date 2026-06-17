# 🐦 Twitter Sentiment Analysis — NLP + ANN

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview
A brand wants to monitor Twitter sentiment at scale. This project builds a full NLP pipeline — text cleaning, TF-IDF vectorization, and a Logistic Regression vs ANN comparison — ending in a deployable `predict_sentiment()` function.

## Dataset
- Sentiment140 (Twitter Dataset) — [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Sampled 50,000 tweets from the full 1.6M for tractable local training

## Tech Stack
Python · Pandas · NumPy · NLTK (stopwords, stemming) · Scikit-learn (TF-IDF, Logistic Regression, MLPClassifier/ANN)

## Approach
1. Sampled 50k tweets and mapped sentiment labels to binary (positive/negative)
2. Cleaned tweet text — removed URLs and mentions, stripped punctuation, removed stopwords, applied stemming
3. Visualized the top 15 most frequent words in positive vs negative tweets
4. Vectorized cleaned text with TF-IDF (10,000 features, unigrams + bigrams)
5. Trained and compared Logistic Regression against an ANN (MLPClassifier, 256→128)
6. Built an end-to-end `predict_sentiment(text)` function — raw text in, sentiment label and confidence out

## Key Results & Insights
- Logistic Regression and the ANN achieve nearly identical AUC (~0.85) — Logistic Regression is the better choice in practice for its speed and simplicity
- Top positive words: love, great, thank, good, happy (stemmed)
- Top negative words: hate, sad, bad, sorry, tired (stemmed)
- Bigrams like "feel good" and "not work" rank among the most predictive features
- The end-to-end pipeline correctly classifies clear positive/negative examples with high confidence (>0.8)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook twitter_sentiment_analysis.ipynb
```

## Project Structure
```
twitter-sentiment-analysis-nlp-ann/
├── notebooks/twitter_sentiment_analysis.ipynb
├── images/  (word_freq.png)
├── requirements.txt
└── README.md
```

## Author
Akshat Kesharwani — [LinkedIn](https://linkedin.com/in/akshat-kesharwani-b0452b346) · [Portfolio](https://akshatkesharwani-info.github.io)
