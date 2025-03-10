# modules/data_collection/liquidity_data.py

import os
import requests
import pandas as pd

def fetch_liquidity_data():
    """
    Fetches liquidity (exchange) data for BTC from CryptoCompare.
    Returns a DataFrame with exchange-level liquidity metrics.
    """
    url = "https://min-api.cryptocompare.com/data/top/exchanges/full?fsym=BTC&tsym=USD"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    # Extract exchange data from the returned JSON
    exchanges_data = data.get("Data", {}).get("Exchanges", [])
    df = pd.DataFrame(exchanges_data)
    return df

def save_liquidity_to_csv(df, filename="liquidity_data.csv"):
    """
    Saves the liquidity data to a CSV file in data/unprocessed/liquidity.
    """
    dir_path = os.path.join("data", "unprocessed", "liquidity")
    os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, filename)
    df.to_csv(file_path, index=False)
    print(f"Liquidity data saved to {file_path}")

def main():
    df = fetch_liquidity_data()
    print(df.head())
    save_liquidity_to_csv(df)

if __name__ == "__main__":
    main()
