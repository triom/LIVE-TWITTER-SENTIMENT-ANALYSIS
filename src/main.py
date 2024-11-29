import pandas as pd
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pyodbc
import uuid
import nltk
from googletrans import Translator
from googletrans import Translator  # Ensure you have googletrans==4.0.0-rc1 installed

# translate all text that is  not in english to have a better performance
def translate_comment(df):
    # Initialize the translator
    translator = Translator()
    df['Text_En'] = df['Text'].apply(lambda comment: translator.translate(comment, dest='en', src='auto').text)
    return df

# Functions

# Data Cleaning and Preparation
def data_cleaning_and_preparation(df, text):
    """
    Cleans the specified text column in a DataFrame:
    - Drops rows with missing or duplicate values in the text column.
    - Converts text to lowercase.
    - Removes special characters, punctuation, numbers, and extra whitespace.
    """
    if text not in df.columns:
        raise ValueError(f"Column '{text}' does not exist in the DataFrame.")

    # Drop rows with missing values in the specified column
    df.dropna(subset=[text], inplace=True)

    # Remove duplicates in the specified column
    df.drop_duplicates(subset=[text], inplace=True)

    # Convert text to lowercase
    df[text] = df[text].str.lower()

    # Remove special characters, punctuation, and numbers
    df[text] = df[text].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

    # Remove extra whitespace
    df[text] = df[text].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    return df


# Sentiment Analysis using Transformers
def sentiment_analysis_transformers(df, text):
    """
    Applies sentiment analysis on the specified text column using a pretrained model.
    - Adds columns for sentiment scores, sentiment labels, and cleaned sentiment labels.
    """
    if text not in df.columns:
        raise ValueError(f"Column '{text}' does not exist in the DataFrame.")

    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    analyser = pipeline("sentiment-analysis", model=model_path)

    # Apply the model and extract scores and labels
    df['scores'] = df[text].apply(lambda text: analyser(text))
    df['Sentiment'] = df['scores'].apply(lambda output: output[0]['label'])
    df['sentiment_score'] = df['scores'].apply(lambda output: output[0]['score'])

    # Map sentiment labels to user-friendly text
    sentiment_map = {
        'positive': 'Positive ❤️',
        'negative': 'Negative 😡',
        'neutral': 'Neutral 💛'
    }
    df['Sentiment'] = df['Sentiment'].map(sentiment_map)

    # Drop intermediate 'scores' column
    df.drop(columns=['scores'], inplace=True)

    return df


# Hashtag Generator
def hashtag_generator(df, text_column):
    """
    Generates hashtags from the text column using a T5 model.
    - Downloads necessary NLTK resources.
    - Adds a 'hashtag' column with generated hashtags.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")

    nltk.download('punkt')

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")

    def generate_tags(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return []  # Return an empty list for invalid text

        inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))  # Unique hashtags
        return tags

    # Apply the hashtag generation function
    df['hashtag'] = df[text_column].apply(generate_tags)

    return df

def emotion_recognition(df, text_column):
    classifier = pipeline("text-classification", model="tasinhoque/roberta-large-go-emotions")
    df['score'] = df[text_column].apply(lambda x: classifier(x.lower()) if isinstance(x, str) else [])
    df['Emotion'] = df['score'].apply(lambda x: x[0]['label'] if x else None)
    df['Percentage'] = df['score'].apply(lambda x: x[0]['score'] if x else 0.0)
    df.drop(columns=['score'], inplace=True)
    return df




# Insert Data into SQL Server
def insert_data(df):
    """
    Inserts DataFrame data into an SQL Server database.
    - Requires specific column names in the DataFrame.
    - Uses pyodbc to connect to SQL Server.
    """
    required_columns = ['tweet_content', 'tweet_date', 'tweet_location', 'hashtag', 'sentiment_score', 'Sentiment', 'id_dist', 'emotion', 'retweets', 'likes']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    # Connect to the SQL Server
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=DESKTOP-8JEBTPG;'
        'DATABASE=tweets;'
        'UID=sa;'
        'PWD=Triomphe&14;'
    )
    cursor = conn.cursor()

    # SQL INSERT query
    insert_query = """
    INSERT INTO [dbo].[tweet] 
        ([tweet_content], [tweet_date], [tweet_location], [hashtag], [sentiment_score], [sentiment], [id_dist], [emotion], [retweet], [like])
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        for index, row in df.iterrows():
            try:
                params = (row['tweet_content'], row['tweet_date'], row['tweet_location'], row['hashtag'],
                        row['sentiment_score'], row['Sentiment'], row['id_dist'], row['emotion'], row['retweets'], row['likes'])
            
                print(f"Executing SQL with params: {params}")
                cursor.execute(insert_query, params)
            except Exception as e:
                print(f"Insertion error {index} : {e}")

        conn.commit()  # Commit if no exception
        print("Data insertion committed successfully.")
    except Exception as e:
        print(f"Erreur globale pendant l'insertion : {e}")
    # close connection
    cursor.close()
    conn.close()



if __name__ == "__main__":
    # df = pd.read_csv("../data/UK_data.csv")
    # Rename the columns in the DataFrame
    # df.rename(columns={
    #     'Text': 'tweet_content',
    #     'Created At': 'tweet_date','Text': 'tweet_content',
    #     'Retweets': 'retweets',
    #     'Likes': 'likes'
    # }, inplace=True)
    # print(df.head())

    # # Data Cleaning and Preparation
    # df_cleaned = data_cleaning_and_preparation(df, 'tweet_content')

    # # Sentiment Analysis using Transformers
    # df_sentiment = sentiment_analysis_transformers(df_cleaned, 'tweet_content')

    # # Hashtag Generation
    # df_hashtags = hashtag_generator(df_sentiment, 'tweet_content')

    # # Emotion Recognition
    # df_emotions = emotion_recognition(df_hashtags, 'tweet_content')

    # # Insert Data into SQL Server
    # insert_data(df_emotions)
    
    df = pd.read_csv("../data/cleaned.csv")
    df['retweets'] = 0
    df['likes'] = 0
    df_cleaned = df.dropna(subset=['sentiment_score'])  # This drops rows where sentiment_score is NaN

    # Ensure 'sentiment_score' is a float and drop rows with invalid values
    df_cleaned['sentiment_score'] = pd.to_numeric(df_cleaned['sentiment_score'], errors='coerce')

    # Drop rows where sentiment_score is NaN after coercion
    df_cleaned = df_cleaned.dropna(subset=['sentiment_score'])

   # Convert tweet_date to datetime format
    df_cleaned['tweet_date'] = pd.to_datetime(df_cleaned['tweet_date'], errors='coerce')

    # Ensure id_dist is a valid UUID
    df_cleaned['id_dist'] = pd.to_numeric(df_cleaned['id_dist'], errors='coerce')

    # Drop any rows with invalid data
    df_cleaned = df_cleaned.dropna(subset=['tweet_date', 'id_dist'])

    print(df_cleaned['sentiment_score'].head())
    insert_data(df_cleaned)
