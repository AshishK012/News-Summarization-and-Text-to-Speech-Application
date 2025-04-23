import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from gtts import gTTS
from io import BytesIO
import pandas as pd
import time
from collections import Counter
from itertools import combinations
import streamlit as st
#from googletrans import Translator
#from googletrans import client


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# --------- NEWS FETCHING FUNCTIONS ---------
def get_google_news_links(company_name):
    search_url = f"https://www.google.com/search?q={company_name}+company+news+-nytimes+-wsj+-ft+-reuters+-pcmag&hl=en&gl=us&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            st.error("Failed to fetch search results")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    articles_list = []
    for g in soup.find_all("div", class_="SoaBEf"):
        a_tag = g.find("a")
        if a_tag and a_tag.get("href"):
            url = a_tag["href"].split("/url?q=")[-1].split("&sa=")[0]
            articles_list.append(url)
    return articles_list if articles_list else []

def get_article_title_and_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else "Title not found"
        if any(term in title.lower() for term in ["sign in", "register", "subscription", "login", "bot protection", "subscribe"]):
            return None, None
        
        article_text = " ".join([p.get_text() for p in soup.find_all("p")])
        title = re.sub(r'[\\/:*?"<>|\n\t]', '', title)
        
        return title, article_text.strip()
    except:
        return None, None

def fetch_articles(company):
    articles_list = get_google_news_links(company)
    if len(articles_list) < 10:
        st.info(f"Only found {len(articles_list)} articles. Expanding search.")
        additional_articles = get_google_news_links(company + " latest")
        for url in additional_articles:
            if url not in articles_list and len(articles_list) < 10:
                articles_list.append(url)
    
    data = []
    used_links = []
    fetched_links = []
    
    for url in articles_list:
        title, text = get_article_title_and_text(url)
        if title and text:
            data.append([title, text])
            used_links.append(url)
            fetched_links.append(url)
        else:
            #st.info(f"Skipping: {url}, fetching a replacement...")
            replacement_articles = get_google_news_links(company + " latest")
            for new_url in replacement_articles:
                if new_url not in used_links:
                    new_title, new_text = get_article_title_and_text(new_url)
                    if new_title and new_text:
                        data.append([new_title, new_text])
                        used_links.append(new_url)
                        fetched_links.append(new_url)
                        break
        
        time.sleep(1)
    
    df = pd.DataFrame(data, columns=['Title', 'Text'])
    
    st.sidebar.subheader("Fetched Article Links")
    for link in fetched_links:
        st.sidebar.write(link)
    
    return df

# --------- TEXT PROCESSING FUNCTIONS ---------
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def replace_contractions(text):
    return contractions.fix(text)

def remove_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    stopwords_set = set(stopwords.words('english'))
    customlist = ['not', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
            "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
            "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stopwords_set = list(set(stopwords_set) - set(customlist))
    
    new_words = []
    for word in words:
        if word not in stopwords_set:
            new_words.append(word)
    return new_words

def lemmatize_list(words):
    lemmatizer = WordNetLemmatizer()
    new_words = []
    for word in words:
        new_words.append(lemmatizer.lemmatize(word, pos='v'))
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_list(words)
    return ' '.join(words)

# --------- ANALYSIS FUNCTIONS ---------
def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)  
    compound = scores['compound']  

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def extractive_summarize(text, max_words=50):
    # Clean and split text into words
    words = nltk.word_tokenize(text.lower())
    stopwords_set = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stopwords_set]

    word_freq = Counter(words)

    # Split original text into chunks (phrases of ~10–15 words)
    tokens = nltk.word_tokenize(text)
    chunks = [' '.join(tokens[i:i+30]) for i in range(0, len(tokens), 30)]

    # Score chunks based on important word frequency
    chunk_scores = []
    for chunk in chunks:
        chunk_words = nltk.word_tokenize(chunk.lower())
        score = sum(word_freq.get(word, 0) for word in chunk_words)
        chunk_scores.append((chunk, score))

    # Sort and select best chunks until we hit max_words
    chunk_scores.sort(key=lambda x: x[1], reverse=True)

    summary = []
    total_words = 0
    for chunk, _ in chunk_scores:
        chunk_word_count = len(nltk.word_tokenize(chunk))
        if total_words + chunk_word_count <= max_words:
            summary.append(chunk)
            total_words += chunk_word_count
        if total_words >= max_words:
            break

    return ' '.join(summary)


def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    return [feature_names[i] for i in scores.argsort()[-top_n:][::-1]]

# --------- TEXT-TO-SPEECH FUNCTION ---------
from deep_translator import GoogleTranslator

def translate_to_hindi(text):
    try:
        return GoogleTranslator(source='auto', target='hi').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text



def text_to_speech_hindi(text):
    tts = gTTS(text, lang="hi")
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer
