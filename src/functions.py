import re
import os
import pandas as pd
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer


# data cleaning and prep
def data_cleaning_and_preparation(df, text_column):
    # 1. Drop rows with missing values in text columns
    df.dropna(subset=[text_column], inplace=True)

    # 2. Remove duplicates
    df.drop_duplicates(subset=[text_column], inplace=True)

    # 3. Convert text to lowercase
    df[text_column] = df[text_column].str.lower()

    # 4. Remove special characters, punctuation, and numimbers
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # 5. Remove extra whitespace
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # 7. Check the cleaned data
    print(df.head())


def sentiment_analysis_transformers(df):
    
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    analyser = pipeline("sentiment-analysis", model=model_path)
    df['scores'] = df['Text'].apply(lambda text: analyser (text.lower()))
    df['Sentiment'] = df['scores'].apply(lambda output: output[0]['label']) 
    df['Sentiment'] = df['Sentiment'].apply(lambda x: 'Positive ❤️' if x == 'positive' else ('Negative 😡' if x == 'negative' else 'Neutral 💛')) 
    
    df =  df.drop(columns = ["scores"])
    return df





