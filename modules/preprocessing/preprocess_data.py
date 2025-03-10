#!/usr/bin/env python3
"""
Advanced Database-Only Preprocessing for Binance OHLCV

This script is designed for profitable AI quant trading using BTC and BTC futures data.
Research shows that financial data in these markets is heavy-tailed, noisy, and non-stationary.
Thus, this script implements robust techniques including:
  - Rigorous data validation and logging
  - Adaptive outlier removal (Z-score or IQR-based)
  - Advanced normalization options (minmax, standard, robust, and log-robust)
  - Comprehensive cleaning (duplicates, missing values, and sorting by timestamp)
  
The processed data is saved into a new table in data/processed/my_trading_data.db.

Usage:
  !python modules/preprocessing/preprocessing.py
"""

import os
import sqlite3
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

#######################
# 1) LOADING RAW DATA
#######################

def load_raw_data_from_db(db_name="my_trading_data.db", table_name="ohlcv_raw") -> pd.DataFrame:
    """
    Load raw OHLCV data from a local SQLite database.
    By default:
      - DB: data/unprocessed/my_trading_data.db
      - Table: 'ohlcv_raw'
    """
    db_path = os.path.join("data", "unprocessed", db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Could not find raw DB at: {db_path}")
    
    logging.info(f"Loading raw data from DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    # Ensure 'timestamp' is in datetime format
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    logging.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    logging.info(f"Data summary before cleaning:\n{df.describe(include='all').to_dict()}")
    return df

####################
# 2) CLEANING STEPS
####################

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    init_count = len(df)
    df = df.drop_duplicates()
    removed = init_count - len(df)
    if removed > 0:
        logging.info(f"Removed {removed} duplicate rows.")
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using forward-fill followed by backward-fill.
    """
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logging.info(f"Found {missing_before} missing values; applying ffill then bfill.")
        df = df.fillna(method='ffill').fillna(method='bfill')
        missing_after = df.isnull().sum().sum()
        logging.info(f"Missing values after filling: {missing_after}")
    return df

def remove_outliers_zscore(df: pd.DataFrame, z_thresh=5.0, columns=None) -> pd.DataFrame:
    """
    Remove rows with an absolute Z-score > z_thresh in specified columns.
    Default columns: open, high, low, close, volume.
    """
    if columns is None:
        possible = ["open", "high", "low", "close", "volume"]
        columns = [c for c in possible if c in df.columns]
    
    if not columns:
        logging.info("No numeric columns available for Z-score outlier removal; skipping.")
        return df

    logging.info(f"Removing outliers using Z-score in {columns} with threshold {z_thresh}...")
    numeric_df = df[columns].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    mask = (numeric_df.abs() <= z_thresh).all(axis=1)
    removed = len(df) - mask.sum()
    if removed > 0:
        logging.info(f"Z-score removal: Removed {removed} outlier rows.")
    return df[mask].reset_index(drop=True)

def remove_outliers_iqr(df: pd.DataFrame, multiplier=1.5, columns=None) -> pd.DataFrame:
    """
    Remove rows outside the [Q1 - multiplier*IQR, Q3 + multiplier*IQR] range.
    """
    if columns is None:
        possible = ["open", "high", "low", "close", "volume"]
        columns = [c for c in possible if c in df.columns]
    
    if not columns:
        logging.info("No numeric columns available for IQR outlier removal; skipping.")
        return df

    logging.info(f"Removing outliers using IQR in {columns} with multiplier {multiplier}...")
    mask = pd.Series(True, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        col_mask = df[col].between(lower_bound, upper_bound)
        mask &= col_mask
        removed_col = (~col_mask).sum()
        if removed_col > 0:
            logging.info(f"IQR removal: Removed {removed_col} outliers in column '{col}'.")
    return df[mask].reset_index(drop=True)

def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the DataFrame by 'timestamp' in ascending order.
    """
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        logging.info("Data sorted by timestamp.")
    else:
        logging.warning("Timestamp column not found; skipping sort.")
    return df

#######################
# 3) NORMALIZATION
#######################

def normalize_data(df: pd.DataFrame, method="none", columns=None) -> pd.DataFrame:
    """
    Normalize numeric columns using one of several methods:
      - "none": No normalization.
      - "minmax": Scale values to [0, 1].
      - "standard": Z-score normalization.
      - "robust": Subtract median and divide by IQR.
      - "log_robust": Log-transform then robust scaling.
    
    :param columns: List of columns to normalize (auto-detected if None).
    """
    if method == "none":
        logging.info("No normalization applied.")
        return df

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in columns:
            columns.remove('timestamp')

    if not columns:
        logging.info("No numeric columns found for normalization; skipping.")
        return df

    if method == "minmax":
        logging.info("Applying min-max normalization...")
        for col in columns:
            min_val, max_val = df[col].min(), df[col].max()
            if max_val - min_val > 1e-12:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                logging.warning(f"Column {col} has negligible range; setting to 0.")
                df[col] = 0.0
    elif method == "standard":
        logging.info("Applying standard (z-score) normalization...")
        for col in columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 1e-12:
                df[col] = (df[col] - mean_val) / std_val
            else:
                logging.warning(f"Column {col} has near-zero std; setting to 0.")
                df[col] = 0.0
    elif method == "robust":
        logging.info("Applying robust normalization (median and IQR)...")
        for col in columns:
            median_val = df[col].median()
            IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
            if IQR > 1e-12:
                df[col] = (df[col] - median_val) / IQR
            else:
                logging.warning(f"Column {col} has near-zero IQR; setting to 0.")
                df[col] = 0.0
    elif method == "log_robust":
        logging.info("Applying log-robust normalization: log-transform then robust scaling...")
        for col in columns:
            # Ensure data is strictly positive by shifting if necessary
            shift = 0
            if (df[col] <= 0).any():
                shift = abs(df[col].min()) + 1e-6
                logging.info(f"Shifting column {col} by {shift:.6f} to allow log transform.")
            df[col] = np.log(df[col] + shift)
            # Now apply robust scaling
            median_val = df[col].median()
            IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
            if IQR > 1e-12:
                df[col] = (df[col] - median_val) / IQR
            else:
                logging.warning(f"Column {col} has near-zero IQR after log transform; setting to 0.")
                df[col] = 0.0
    else:
        logging.warning(f"Unknown normalization method '{method}'; no normalization applied.")
    return df

###############################
# 4) SAVE PROCESSED DATA TO DB
###############################

def save_processed_to_db(df: pd.DataFrame, db_name="my_trading_data.db", table_name="ohlcv_processed"):
    """
    Save the processed DataFrame to a local SQLite database.
    Default: data/processed/my_trading_data.db, table 'ohlcv_processed'.
    """
    db_path = os.path.join("data", "processed", db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    logging.info(f"Saving processed data to DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logging.info("Processed data successfully stored in the DB.")

#######################
# 5) MAIN PIPELINE
#######################

def preprocess_ohlcv(
    db_name="my_trading_data.db",
    table_name_raw="ohlcv_raw",
    remove_outliers_flag=True,
    outlier_method="iqr",       # Options: "zscore" or "iqr"
    z_thresh=5.0,               # For Z-score method
    iqr_multiplier=1.5,         # For IQR method
    normalization_method="log_robust",  # Options: "none", "minmax", "standard", "robust", "log_robust"
    table_name_processed="ohlcv_processed"
):
    """
    End-to-end preprocessing pipeline to:
      - Load raw OHLCV data (default table 'ohlcv_raw' in data/unprocessed/).
      - Remove duplicates and fill missing values.
      - Optionally remove outliers using either the Z-score or IQR method.
      - Sort by timestamp.
      - Apply advanced normalization tailored for heavy-tailed financial data.
      - Save the processed data to the specified DB/table.
    """
    try:
        # 1) Load raw data and log summary statistics
        df_raw = load_raw_data_from_db(db_name=db_name, table_name=table_name_raw)

        # 2) Clean data: remove duplicates and fill missing values
        df = remove_duplicates(df_raw)
        df = fill_missing_values(df)
        
        # 3) Outlier removal (choose method based on market research insights)
        if remove_outliers_flag:
            if outlier_method == "zscore":
                df = remove_outliers_zscore(df, z_thresh=z_thresh)
            elif outlier_method == "iqr":
                df = remove_outliers_iqr(df, multiplier=iqr_multiplier)
            else:
                logging.warning(f"Unknown outlier method: {outlier_method}; skipping outlier removal.")
        
        # 4) Ensure data is sorted by timestamp
        df = sort_by_timestamp(df)
        
        # 5) Apply advanced normalization/scaling
        df = normalize_data(df, method=normalization_method)
        
        # 6) Log final summary
        logging.info(f"Final processed data: {df.shape[0]} rows; summary:\n{df.describe(include='all').to_dict()}")
        
        # 7) Save the processed data to the output DB/table
        save_processed_to_db(df, db_name=db_name, table_name=table_name_processed)
        logging.info("Preprocessing pipeline complete. Data stored in the processed DB table.")
    except Exception as e:
        logging.error("Error during preprocessing: " + str(e))
        raise

def main():
    """
    Example usage:
      - Load raw data from data/unprocessed/my_trading_data.db (table 'ohlcv_raw').
      - Remove duplicates, fill missing values.
      - Remove outliers using the IQR method.
      - Sort by timestamp.
      - Apply log-robust normalization to handle heavy tails.
      - Save the result to data/processed/my_trading_data.db (table 'ohlcv_processed').
    """
    preprocess_ohlcv(
        db_name="my_trading_data.db",
        table_name_raw="ohlcv_raw",
        remove_outliers_flag=True,
        outlier_method="iqr",         # Choose "zscore" or "iqr"
        z_thresh=5.0,                 # For Z-score method (if used)
        iqr_multiplier=1.5,           # For IQR method
        normalization_method="log_robust",  # Options: "none", "minmax", "standard", "robust", "log_robust"
        table_name_processed="ohlcv_processed"
    )

if __name__ == "__main__":
    main()
