# modules/data_collection/on_chain_data.py

import os
import requests
import pandas as pd

def fetch_onchain_data():
    """
    Fetches on-chain data from the Blockchain.info API.
    Returns a DataFrame with network statistics.
    """
    url = "https://api.blockchain.info/stats?format=json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    # Convert the dictionary into a DataFrame with one row
    df = pd.DataFrame([data])
    return df

def save_onchain_to_csv(df, filename="onchain_data.csv"):
    """
    Saves the on-chain data to a CSV file in data/unprocessed/onchain.
    """
    dir_path = os.path.join("data", "unprocessed", "onchain")
    os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, filename)
    df.to_csv(file_path, index=False)
    print(f"On-chain data saved to {file_path}")

def main():
    df = fetch_onchain_data()
    print(df.head())
    save_onchain_to_csv(df)

if __name__ == "__main__":
    main()
