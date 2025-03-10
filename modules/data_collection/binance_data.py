#!/usr/bin/env python3
"""
Binance Data Collection + Database Storage (DB-Only Version)

This script:
  1. Fetches historical OHLCV data from Binance using ccxt in incremental chunks.
  2. Saves the raw data exclusively to a local SQLite database table (default: 'ohlcv_raw').

We store the timestamp in UTC and retry on errors with exponential backoff if needed.
"""

import os
import time
import logging
import sqlite3
import pandas as pd
import ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def fetch_binance_ohlcv(symbol="BTC/USDT", timeframe="15m", since=None, max_retries=5):
    """
    Fetch historical OHLCV data from Binance in incremental chunks, 
    handling network errors with retries.
    
    :param symbol: Trading pair (e.g. "BTC/USDT")
    :param timeframe: Candlestick timeframe (e.g. "15m", "1h", "1d")
    :param since: Timestamp (ms) for the start of data. If None, fetches most recent data.
    :param max_retries: Max attempts on fetch errors
    
    :return: pd.DataFrame with columns [timestamp, open, high, low, close, volume],
             where 'timestamp' is UTC datetime.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    all_ohlcv = []
    limit = 1000

    while True:
        ohlcv = None
        # Attempt up to max_retries
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                break
            except Exception as e:
                logging.warning(f"Fetch error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # exponential backoff

        if ohlcv is None:
            raise ConnectionError(f"Max retries ({max_retries}) exceeded while fetching data from Binance.")

        if not ohlcv:
            # No more data returned
            break

        all_ohlcv.extend(ohlcv)

        # Update 'since' to 1ms after the last candle
        new_since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit or new_since == since:
            # Reached end or no progress
            break
        since = new_since

    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def save_to_database(df, db_name="my_trading_data.db", table_name="ohlcv_raw"):
    """
    Save the fetched DataFrame to a local SQLite database. 
    By default, store in data/unprocessed/my_trading_data.db.
    
    :param df: DataFrame with columns [timestamp, open, high, low, close, volume].
    :param db_name: SQLite DB filename
    :param table_name: table name in the DB
    """
    db_path = os.path.join("data", "unprocessed", db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    # If you prefer incremental updates, you can switch if_exists='append'
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

    logging.info(f"Data saved to SQLite DB: {db_path} (table: {table_name})")


def main():
    """
    Example usage:
      1) Fetch from Binance (BTC/USDT, 15m) starting 2017-01-01
      2) Save only to a local SQLite DB (my_trading_data.db, table 'ohlcv_raw').
      
    No CSV saving. DB only.
    """
    # 1) Fetch data
    start_timestamp = int(pd.Timestamp("2017-01-01").timestamp() * 1000)
    symbol, timeframe = "BTC/USDT", "15m"
    df = fetch_binance_ohlcv(symbol, timeframe, since=start_timestamp)

    # 2) Save to local SQLite DB
    save_to_database(df, db_name="my_trading_data.db", table_name="ohlcv_raw")


if __name__ == "__main__":
    main()
