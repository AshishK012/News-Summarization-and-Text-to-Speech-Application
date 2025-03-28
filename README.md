<!-- README.md -->
<!DOCTYPE html>
<html>
<head>
    <title>Company News Analysis App</title>
</head>
<body>
    <h1>Company News Analysis App</h1>
    <p>This repository contains a Streamlit app and a FastAPI backend for fetching, analyzing, and summarizing news articles related to a given company.</p>
    <p>The application includes features such as sentiment analysis, topic extraction, comparative summaries, Hindi translation, and text-to-speech functionality.</p>
    
    <h2>Features</h2>
    <ul>
        <li>Fetches latest news articles from Google News.</li>
        <li>Performs sentiment analysis on the article content.</li>
        <li>Extracts keywords and topics using TF-IDF.</li>
        <li>Generates short extractive summaries.</li>
        <li>Provides comparative sentiment analysis between different articles.</li>
        <li>Supports translation of summaries to Hindi.</li>
        <li>Converts Hindi summaries to speech (Text-to-Speech).</li>
        <li>FastAPI backend to provide API endpoints for analysis.</li>
    </ul>
    
    <h2>Project Structure</h2>
    <pre>
    📺 company-news-analysis
    │── 📂 app
    │   ├── app.py (Streamlit front-end)
    │   ├── requirements.txt (Dependencies)
    │   ├── README.md (Project documentation)
    │   ├── utils.py (Utility functions)
    │── 📂 api
    │   ├── api.py (FastAPI backend)
    │── 📂 nltk_data (NLTK pre-downloaded data)
    </pre>
    
    <h2>Live Demo</h2>
    <p>Try out the application:</p>
    <ul>
        <li><a href="https://huggingface.co/spaces/Ashu804/company-news-analysis" target="_blank">Hugging Face Space</a></li>
        <li><a href="https://news-summarization-and-text-to-speech-application-vhphruajhtnz.streamlit.app/" target="_blank">Streamlit Deployment</a></li>
    </ul>
</body>
</html>
