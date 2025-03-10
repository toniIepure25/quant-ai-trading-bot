# modules/data_collection/sentiment_data.py

import os
import requests
import pandas as pd
from datetime import datetime

def fetch_sentiment_data():
    """
    Fetches cryptocurrency sentiment data from the Alternative.me Fear & Greed Index API.
    Returns a DataFrame.
    """
    url = "https://api.alternative.me/fng/?limit=1"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    data = response.json()
    
    # The API returns a JSON with a "data" key containing a list of entries
    sentiment_list = data.get("data", [])
    df = pd.DataFrame(sentiment_list)
    
    # Convert the 'timestamp' field from UNIX seconds to a readable datetime, if present
    # if 'timestamp' in df.columns:
    #     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    if 'timestamp' in df.columns:
    # Convert to float first, then to datetime using UNIX seconds
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')

    return df

def save_sentiment_to_csv(df, filename="sentiment_data.csv"):
    """
    Saves the sentiment DataFrame as a CSV file in data/unprocessed/sentiment.
    """
    dir_path = os.path.join("data", "unprocessed", "sentiment")
    os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, filename)
    df.to_csv(file_path, index=False)
    print(f"Sentiment data saved to {file_path}")

def main():
    df = fetch_sentiment_data()
    print(df.head())
    save_sentiment_to_csv(df)

if __name__ == "__main__":
    main()
