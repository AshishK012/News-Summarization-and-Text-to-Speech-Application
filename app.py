import streamlit as st
import pandas as pd
from utils import fetch_articles, get_sentiment, extractive_summarize, extract_keywords, text_to_speech_hindi
from  utils import translate_to_hindi,strip_html,replace_contractions,remove_numbers,normalize
import nltk
from itertools import combinations

import nltk
import os

# Set NLTK data path to the repo folder
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))


def main():
    st.title("Company News Analysis App")
    
    st.sidebar.header("Company News Analysis")
    company_name = st.sidebar.text_input("Enter Company Name:")
    
    if st.sidebar.button("Analyze News"):
        if company_name:
            df = fetch_articles(company_name)
            
            df['Text'] = df['Text'].apply(strip_html)
            df['Title'] = df['Title'].apply(strip_html)
            df['Text'] = df['Text'].apply(replace_contractions)
            df['Title'] = df['Title'].apply(replace_contractions)
            df['Text'] = df['Text'].apply(remove_numbers)
            df['Title'] = df['Title'].apply(remove_numbers)
            
            df['Text'] = df.apply(lambda row: nltk.word_tokenize(row['Text']), axis=1)
            df['Title'] = df.apply(lambda row: nltk.word_tokenize(row['Title']), axis=1)
            
            df['Text'] = df.apply(lambda row: normalize(row['Text']), axis=1)
            df['Title'] = df.apply(lambda row: normalize(row['Title']), axis=1)
            
            df['Sentiment'] = df['Text'].apply(get_sentiment)
            df["Summary"] = df["Text"].apply(extractive_summarize)
            
            if len(df) >= 2:
                comparisons = []
                topic_overlaps = []
                
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
                
                st.subheader("Comparative Analysis")
                st.json(analysis)
                
                if not df.empty:
                    for i, row in df.iterrows():
                        title = row['Title']
                        st.write(f"**{title}**")
                        summary_text = row['Text'][:500]
                        hindi_text = translate_to_hindi(summary_text)  # Translating to Hindi
        
                        hindi_speech = text_to_speech_hindi(hindi_text)  # Converting to Hindi speech
                        st.audio(hindi_speech, format='audio/mp3')
            else:
                st.warning("Not enough articles for comparative analysis.")
        else:
            st.error("Please enter a company name.")

if __name__ == "__main__":
    main()
