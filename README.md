Company News Analysis App

This repository contains a Streamlit app and a FastAPI backend for fetching, analyzing, and summarizing news articles related to a given company. The application includes features such as sentiment analysis, topic extraction, comparative summaries, Hindi translation, and text-to-speech functionality.

Features

Fetches latest news articles from Google News.

Performs sentiment analysis on the article content.

Extracts keywords and topics using TF-IDF.

Generates short extractive summaries.

Provides comparative sentiment analysis between different articles.

Supports translation of summaries to Hindi.

Converts Hindi summaries to speech (Text-to-Speech).

FastAPI backend to provide API endpoints for analysis.

Project Structure

📂 company-news-analysis
│── 📂 app
│   ├── app.py (Streamlit front-end)
│   ├── requirements.txt (Dependencies)
│   ├── README.md (Project documentation)
│   ├── utils.py (Utility functions)
│── 📂 api
│   ├── api.py (FastAPI backend)
│── 📂 nltk_data (NLTK pre-downloaded data)

