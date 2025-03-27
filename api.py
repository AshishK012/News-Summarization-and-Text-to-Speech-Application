# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json


from utils import fetch_articles, strip_html, replace_contractions, remove_numbers, normalize, get_sentiment, extractive_summarize, extract_keywords
import nltk
from itertools import combinations
import time
import pandas as pd



app = FastAPI(title="Company News Analysis API")

class CompanyRequest(BaseModel):
    company_name: str

@app.post("/analyze_news")
async def analyze_news(request: CompanyRequest):
    company_name = request.company_name
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required.")
    
    # Fetch articles
    df = fetch_articles(company_name)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No articles found for the given company.")
    
    # Process articles similar to your streamlit main() logic (without UI components)
    df['Text'] = df['Text'].apply(strip_html)
    df['Title'] = df['Title'].apply(strip_html)
    df['Text'] = df['Text'].apply(replace_contractions)
    df['Title'] = df['Title'].apply(replace_contractions)
    df['Text'] = df['Text'].apply(remove_numbers)
    df['Title'] = df['Title'].apply(remove_numbers)
    
    # Tokenize, normalize, and compute sentiment
    df['Text'] = df.apply(lambda row: nltk.word_tokenize(row['Text']), axis=1)
    df['Title'] = df.apply(lambda row: nltk.word_tokenize(row['Title']), axis=1)
    df['Text'] = df.apply(lambda row: normalize(row['Text']), axis=1)
    df['Title'] = df.apply(lambda row: normalize(row['Title']), axis=1)
    
    df['Sentiment'] = df['Text'].apply(get_sentiment)
    df["Summary"] = df["Text"].apply(lambda text: extractive_summarize(text))
    
    comparisons = []
    topic_overlaps = []
    
    if len(df) >= 2:
        for (idx1, article_1), (idx2, article_2) in combinations(df.iterrows(), 2):
            article_1_text = article_1["Summary"]
            article_2_text = article_2["Summary"]
            
            keywords_1 = extract_keywords(article_1_text)
            keywords_2 = extract_keywords(article_2_text)
            
            common_keywords = set(keywords_1).intersection(set(keywords_2))
            unique_keywords_1 = set(keywords_1) - common_keywords
            unique_keywords_2 = set(keywords_2) - common_keywords
            
            comparisons.append({
                "Summary 1": article_1["Summary"],
                "Summary 2": article_2["Summary"],
                "Sentiment Comparison": f"{article_1.get('Sentiment', 'Neutral')} vs {article_2.get('Sentiment', 'Neutral')}"
            })
            
            topic_overlaps.append({ 
                "Article 1": article_1["Title"],
                "Article 2": article_2["Title"],
                "Common Topics": list(common_keywords),
                "Unique Topics in Article 1": list(unique_keywords_1),
                "Unique Topics in Article 2": list(unique_keywords_2)
            })
    
    analysis = {
        "Company": company_name,
        "Articles": [
            {
                "Title": row["Title"],
                "Summary": " ".join(row.get("Text", "").split()[:7]) + "...",
                "Sentiment": row.get("Sentiment", "Neutral"),
                "Topics": extract_keywords(row["Text"])
            }
            for _, row in df.iterrows()
        ],
        "Comparative Sentiment Score": {
            "Sentiment Distribution": {
                "Positive": int((df["Sentiment"] == "Positive").sum()),
                "Negative": int((df["Sentiment"] == "Negative").sum()),
                "Neutral": int((df["Sentiment"] == "Neutral").sum())
            }
        },
        "Coverage Differences": comparisons,
        "Topic Overlap": topic_overlaps,
        "Final Sentiment Analysis": "Overall sentiment leans towards " + (
            "Positive" if (df["Sentiment"] == "Positive").sum() > (df["Sentiment"] == "Negative").sum()
            else "Negative" if (df["Sentiment"] == "Negative").sum() > (df["Sentiment"] == "Positive").sum()
            else "Neutral"
        )
    }
    
    # Return the analysis as JSON
    return analysis

if __name__ == "__main__":
    # Run the API with: uvicorn api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)


