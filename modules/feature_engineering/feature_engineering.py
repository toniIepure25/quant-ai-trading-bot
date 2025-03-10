#!/usr/bin/env python3
"""
Advanced Feature Engineering Module for BTC/USDT OHLCV Data (DB-Friendly)

This script is designed based on research insights for profitable AI quant trading.
It performs advanced feature engineering on processed OHLCV data by:
  - Loading data from either a CSV (data/processed/market/) or a local SQLite DB.
  - Computing a broad set of features including:
       • Basic technical indicators: Log Returns, SMA, EMA, RSI, Bollinger Bands, MACD, ATR, rolling volatility, stochastic oscillator.
       • Additional statistics: Rolling skewness and kurtosis of log returns.
       • Volume/trend indicators: OBV and ADX.
       • Advanced metrics: Hurst Exponent, GARCH/EGARCH volatility.
       • Multiscale analysis: Fourier transform and Wavelet transform (detail energy).
       • Fractal & complexity measures: Recurrence Rate, Lyapunov Exponent, Higuchi Fractal Dimension, DFA exponent, CCI.
       • Regime detection: A new regime label computed via KMeans clustering on a rolling volatility measure.
       • (Optional) Lagged features.
       • Robust normalization tailored for heavy‑tailed financial data (e.g. log‑robust scaling).
  - Saving the enriched feature set to a new CSV and/or a table in the SQLite DB.

Usage:
  !python advanced_feature_engineering.py
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert
from scipy import stats
from sklearn.cluster import KMeans
from joblib import Memory, Parallel, delayed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
memory = Memory(location='./cache', verbose=0)

#############################################
# 1) DATA LOADING AND SAVING FUNCTIONS
#############################################
def load_processed_data_from_db(db_name="my_trading_data.db", table_name="ohlcv_processed") -> pd.DataFrame:
    db_path = os.path.join("data", "processed", db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Processed DB not found: {db_path}")
    logging.info(f"Loading processed data from DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    logging.info(f"Loaded {df.shape[0]} rows from DB.")
    return df

def save_features_to_db(df: pd.DataFrame, db_name="my_trading_data.db", table_name="ohlcv_features"):
    db_path = os.path.join("data", "processed", db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    logging.info(f"Saving feature-enriched data to DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logging.info("Feature data successfully stored in the DB.")

def load_processed_data_from_csv(csv_filename="BTC_USDT_15m_processed.csv") -> pd.DataFrame:
    csv_path = os.path.join("data", "processed", "market", csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")
    logging.info(f"Loading processed data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    logging.info(f"Loaded {df.shape[0]} rows from CSV.")
    return df

def save_features_to_csv(df: pd.DataFrame, csv_filename="BTC_USDT_15m_features.csv"):
    output_dir = os.path.join("data", "processed", "market")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, csv_filename)
    df.to_csv(output_path, index=False)
    logging.info(f"Feature-enriched data saved to CSV: {output_path}")

#############################################
# 2) ADVANCED NORMALIZATION FUNCTIONS
#############################################
def robust_normalize(df: pd.DataFrame, columns=None, method="log_robust"):
    """
    Normalize numeric columns using robust methods:
      - "robust": (x - median) / IQR.
      - "log_robust": Shift values to be strictly positive (if needed), apply log, then robust scale.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in columns:
            columns.remove('timestamp')
    for col in columns:
        if method == "log_robust":
            if (df[col] <= 0).any():
                shift = abs(df[col].min()) + 1e-6
                logging.info(f"Column '{col}': Shifting by {shift:.6f} to allow log transform.")
                df[col] = df[col] + shift
            df[col] = np.log(df[col])
        median = df[col].median()
        IQR = df[col].quantile(0.75) - df[col].quantile(0.25) + 1e-6
        df[col] = (df[col] - median) / IQR
        logging.info(f"Column '{col}' normalized using {method} scaling.")
    return df

#############################################
# 3) ADDITIONAL STATISTICAL FEATURES
#############################################
def add_rolling_stats(df: pd.DataFrame, window=14) -> pd.DataFrame:
    """
    Add rolling skewness and kurtosis of log returns.
    """
    df[f'rolling_skew_{window}'] = df['log_return'].rolling(window=window, min_periods=1).apply(lambda x: stats.skew(x), raw=True)
    df[f'rolling_kurtosis_{window}'] = df['log_return'].rolling(window=window, min_periods=1).apply(lambda x: stats.kurtosis(x), raw=True)
    logging.info(f"Rolling skewness and kurtosis computed for window {window}.")
    return df

#############################################
# 4) REGIME DETECTION FEATURES
#############################################
def add_regime_labels(df: pd.DataFrame, window=50, n_clusters=3) -> pd.DataFrame:
    """
    Compute rolling volatility and cluster its values using KMeans to generate regime labels.
    The new column 'vol_regime' indicates the regime.
    """
    df['rolling_vol'] = df['log_return'].rolling(window=window, min_periods=window).std()
    valid_idx = df['rolling_vol'].dropna().index
    vol_values = df.loc[valid_idx, 'rolling_vol'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    regime_labels = kmeans.fit_predict(vol_values)
    df['vol_regime'] = np.nan
    df.loc[valid_idx, 'vol_regime'] = regime_labels
    # Forward fill any missing regime labels
    df['vol_regime'].fillna(method='ffill', inplace=True)
    logging.info(f"Market regime labels computed using KMeans with {n_clusters} clusters.")
    return df

#############################################
# 5) EXISTING & ADVANCED FEATURE FUNCTIONS
#############################################
def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    logging.info("Log returns computed.")
    return df

def add_moving_averages(df: pd.DataFrame, windows=[10, 20, 50, 100]) -> pd.DataFrame:
    for w in windows:
        df[f'sma_{w}'] = df['close'].rolling(window=w, min_periods=1).mean()
        logging.info(f"SMA ({w}) computed.")
    return df

def add_ema(df: pd.DataFrame, windows=[10, 20, 50]) -> pd.DataFrame:
    for w in windows:
        df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
        logging.info(f"EMA ({w}) computed.")
    return df

def add_rsi(df: pd.DataFrame, window=14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    logging.info("RSI computed.")
    return df

def add_bollinger_bands(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
    sma = df['close'].rolling(window, min_periods=1).mean()
    rolling_std = df['close'].rolling(window, min_periods=1).std()
    df['bollinger_middle'] = sma
    df['bollinger_upper'] = sma + num_std * rolling_std
    df['bollinger_lower'] = sma - num_std * rolling_std
    logging.info("Bollinger Bands computed.")
    return df

def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    logging.info("MACD computed.")
    return df

def add_atr(df: pd.DataFrame, window=14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - df['close'].shift(1)).abs()
    low_prev_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=window, min_periods=1).mean()
    logging.info("ATR computed.")
    return df

def add_rolling_volatility(df: pd.DataFrame, windows=[14, 30]) -> pd.DataFrame:
    for w in windows:
        df[f'volatility_{w}'] = df['log_return'].rolling(window=w, min_periods=1).std()
        logging.info(f"Rolling volatility ({w}) computed.")
    return df

def add_stochastic_oscillator(df: pd.DataFrame, window=14, smooth_k=3) -> pd.DataFrame:
    low_min = df['low'].rolling(window, min_periods=1).min()
    high_max = df['high'].rolling(window, min_periods=1).max()
    df['stoch_%K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_%D'] = df['stoch_%K'].rolling(window=smooth_k, min_periods=1).mean()
    logging.info("Stochastic Oscillator computed.")
    return df

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    change = df['close'].diff().fillna(0)
    direction = np.where(change > 0, 1, np.where(change < 0, -1, 0))
    df['obv'] = (direction * df['volume']).cumsum()
    logging.info("OBV computed.")
    return df

def add_adx(df: pd.DataFrame, window=14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - df['close'].shift(1)).abs()
    low_prev_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = -df['low'].diff().clip(upper=0)
    plus_dm[plus_dm > tr] = 0
    minus_dm[minus_dm > tr] = 0
    tr_smooth = tr.rolling(window=window, min_periods=1).sum()
    plus_dm_smooth = plus_dm.rolling(window=window, min_periods=1).sum()
    minus_dm_smooth = minus_dm.rolling(window=window, min_periods=1).sum()
    di_plus = 100 * (plus_dm_smooth / tr_smooth)
    di_minus = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan))
    df['adx'] = dx.rolling(window=window, min_periods=1).mean()
    logging.info("ADX computed.")
    return df

def add_hurst_exponent(df: pd.DataFrame) -> pd.DataFrame:
    def compute_hurst(ts):
        lags = range(2, 20)
        tau = [np.std(ts.diff(lag).dropna()) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    try:
        hurst_val = compute_hurst(df['close'].dropna())
        df['hurst_exponent'] = hurst_val
        logging.info(f"Hurst Exponent computed: {hurst_val:.4f}")
    except Exception as e:
        logging.error("Error computing Hurst Exponent: " + str(e))
        df['hurst_exponent'] = np.nan
    return df

def add_garch_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from arch import arch_model
    except ImportError:
        logging.warning("arch package not found; skipping GARCH features.")
        df['garch_vol'] = np.nan
        df['egarch_vol'] = np.nan
        return df
    returns = df['close'].pct_change().dropna() * 100
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off')
        df['garch_vol'] = res.conditional_volatility.mean()
        logging.info("GARCH volatility computed.")
    except Exception as e:
        logging.error(f"Error computing GARCH volatility: {e}")
        df['garch_vol'] = np.nan
    try:
        model_egarch = arch_model(returns, vol='EGarch', p=1, q=1, dist='normal')
        res_egarch = model_egarch.fit(disp='off')
        df['egarch_vol'] = res_egarch.conditional_volatility.mean()
        logging.info("EGARCH volatility computed.")
    except Exception as e:
        logging.error(f"Error computing EGARCH volatility: {e}")
        df['egarch_vol'] = np.nan
    return df

def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    fft_vals = np.fft.fft(df['close'].values)
    fft_freq = np.fft.fftfreq(len(df['close'].values))
    pos_mask = fft_freq > 0
    fft_vals = fft_vals[pos_mask]
    fft_freq = fft_freq[pos_mask]
    amplitude = np.abs(fft_vals)
    if len(amplitude) == 0:
        df['dominant_freq'] = np.nan
        df['dominant_amp'] = np.nan
    else:
        dominant_idx = np.argmax(amplitude)
        df['dominant_freq'] = fft_freq[dominant_idx]
        df['dominant_amp'] = amplitude[dominant_idx]
    logging.info("Fourier features computed.")
    return df

def add_wavelet_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        x = df['close'].dropna().values.astype(float)
        max_length = 10000
        if len(x) > max_length:
            logging.info(f"Downsampling 'close' series from {len(x)} to {max_length} for wavelet computation.")
            x = x[-max_length:]
        coeffs = pywt.wavedec(x, 'db1', level=3)
        energies = [np.sum(np.square(c)) for c in coeffs[1:]]
        df['wavelet_energy'] = np.sum(energies)
        logging.info("Wavelet energy computed.")
    except Exception as e:
        logging.error(f"Error computing wavelet features: {e}")
        df['wavelet_energy'] = np.nan
    return df

def add_rqa_features(df: pd.DataFrame, embedding_dim=2, threshold=None, max_points=2000) -> pd.DataFrame:
    from scipy.spatial.distance import pdist
    x = df['close'].values
    N = len(x)
    if N > max_points:
        logging.info(f"Downsampling 'close' series from {N} to {max_points} for RQA.")
        x = x[-max_points:]
        N = len(x)
    if N < embedding_dim:
        df['recurrence_rate'] = np.nan
        return df
    embedded = np.array([x[i:i+embedding_dim] for i in range(N - embedding_dim + 1)])
    dists = pdist(embedded)
    if threshold is None:
        threshold = np.percentile(dists, 10)
    recurrence_rate = np.sum(dists < threshold) / len(dists)
    df['recurrence_rate'] = recurrence_rate
    logging.info("Recurrence rate computed.")
    return df

def add_lyapunov_exponent(df: pd.DataFrame) -> pd.DataFrame:
    x = df['close'].values
    lags = np.arange(1, 50)
    divergences = [np.mean(np.log(np.abs(x[lag:] - x[:-lag]) + 1e-8)) for lag in lags]
    if len(lags) > 1:
        coeffs = np.polyfit(np.log(lags), divergences, 1)
        lyap_exp = coeffs[0]
    else:
        lyap_exp = np.nan
    df['lyapunov_exp'] = lyap_exp
    logging.info(f"Lyapunov exponent computed: {lyap_exp:.4f}")
    return df

def add_hilbert_features(df: pd.DataFrame) -> pd.DataFrame:
    analytic_signal = hilbert(df['close'])
    df['hilbert_amplitude'] = np.abs(analytic_signal)
    df['hilbert_phase'] = np.angle(analytic_signal)
    logging.info("Hilbert transform features computed.")
    return df

@memory.cache
def higuchi_fd(x, kmax=10):
    N = len(x)
    def compute_L_for_k(k):
        Lk = []
        for m in range(k):
            n_max = int(np.floor((N - m - 1) / k))
            if n_max < 1:
                continue
            Lmk = sum(abs(x[m + i * k] - x[m + (i - 1) * k]) for i in range(1, n_max))
            Lmk = (Lmk * (N - 1)) / (n_max * k)
            Lk.append(Lmk)
        return np.log(np.mean(Lk)) if Lk else None
    results = Parallel(n_jobs=-1)(delayed(compute_L_for_k)(k) for k in range(1, kmax+1))
    L_vals = [r for r in results if r is not None]
    if len(L_vals) < 2:
        return np.nan
    ln_k = np.log(np.arange(1, len(L_vals) + 1))
    coeffs = np.polyfit(ln_k, L_vals, 1)
    return coeffs[0]

def add_higuchi_fd(df: pd.DataFrame, kmax=10) -> pd.DataFrame:
    try:
        fd = higuchi_fd(df['close'].values, kmax=kmax)
        df['higuchi_fd'] = fd
        logging.info(f"Higuchi Fractal Dimension computed: {fd:.4f}")
    except Exception as e:
        logging.error(f"Error computing Higuchi FD: {e}")
        df['higuchi_fd'] = np.nan
    return df

@memory.cache
def dfa_exponent(ts, scale_min=5, scale_max=50, scale_count=20):
    scales = np.linspace(scale_min, scale_max, scale_count, dtype=int)
    def compute_fluctuation(s):
        if s >= len(ts):
            return None
        segments = len(ts) // s
        rms_vals = [
            np.sqrt(np.mean((seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1), np.arange(len(seg))))**2))
            for seg in (ts[i*s:(i+1)*s] for i in range(segments))
        ]        
        return np.mean(rms_vals) if rms_vals else None
    results = Parallel(n_jobs=-1)(delayed(compute_fluctuation)(s) for s in scales)
    fluctuations = [r for r in results if r is not None]
    if len(fluctuations) < 2:
        return np.nan
    valid_scales = scales[:len(fluctuations)]
    coeffs = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)
    return coeffs[0]

def add_dfa_exponent(df: pd.DataFrame) -> pd.DataFrame:
    try:
        exponent = dfa_exponent(df['close'].values)
        df['dfa_exponent'] = exponent
        logging.info(f"DFA exponent computed: {exponent:.4f}")
    except Exception as e:
        logging.error(f"Error computing DFA exponent: {e}")
        df['dfa_exponent'] = np.nan
    return df

def add_cci(df: pd.DataFrame, window=20) -> pd.DataFrame:
    TP = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = TP.rolling(window=window, min_periods=1).mean()
    mad = TP.rolling(window=window, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci'] = (TP - sma_tp) / (0.015 * mad)
    logging.info("CCI computed.")
    return df

#############################################
# 6) NEW OPTIONAL FEATURES
#############################################
def add_lagged_features(df: pd.DataFrame, lags=[1, 2, 3]) -> pd.DataFrame:
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    logging.info(f"Lagged features added for lags: {lags}")
    return df

#############################################
# 7) BUILD COMPLETE FEATURE SET
#############################################
def build_features(df: pd.DataFrame, add_extra_lags=True) -> pd.DataFrame:
    """
    Build a comprehensive feature set by sequentially applying:
      - Traditional technical indicators,
      - Rolling statistics (skewness and kurtosis),
      - Regime detection using KMeans on rolling volatility,
      - Multiscale analysis (Fourier and Wavelet transforms),
      - Fractal and complexity measures,
      - Optional lagged features,
      - And robust normalization.
    """
    df = add_log_returns(df)
    df = add_moving_averages(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_rolling_volatility(df)
    df = add_stochastic_oscillator(df)
    df = add_obv(df)
    df = add_adx(df)
    df = add_hurst_exponent(df)
    df = add_garch_features(df)
    df = add_fourier_features(df)
    df = add_wavelet_features(df)
    df = add_rqa_features(df)
    df = add_lyapunov_exponent(df)
    df = add_hilbert_features(df)
    df = add_higuchi_fd(df)
    df = add_dfa_exponent(df)
    df = add_cci(df)
    # Additional statistical measures: rolling skewness and kurtosis
    df = add_rolling_stats(df, window=14)
    # Regime detection based on rolling volatility (window of 50 periods, 3 clusters)
    df = add_regime_labels(df, window=50, n_clusters=3)
    # Apply robust normalization ("log_robust") for heavy-tailed distributions
    df = robust_normalize(df, method="log_robust")
    if add_extra_lags:
        df = add_lagged_features(df, lags=[1, 2, 3])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info("All features built successfully.")
    return df

#############################################
# 8) MAIN PIPELINE
#############################################
def feature_engineering(
    source="db", 
    db_name="my_trading_data.db", 
    table_name_processed="ohlcv_processed", 
    table_name_features="ohlcv_features",
    csv_input="BTC_USDT_15m_processed.csv",
    csv_output="BTC_USDT_15m_features.csv",
    save_to_csv=True,
    save_to_db=True,
    add_extra_lags=True
):
    """
    Main pipeline:
      - Load processed OHLCV data from a DB or CSV.
      - Build the comprehensive feature set.
      - Save the enriched DataFrame to CSV and/or a SQLite DB.
    """
    if source == "db":
        df = load_processed_data_from_db(db_name, table_name_processed)
    elif source == "csv":
        df = load_processed_data_from_csv(csv_input)
    else:
        raise ValueError("source must be 'db' or 'csv'")
    
    df_features = build_features(df, add_extra_lags=add_extra_lags)
    
    if save_to_csv:
        save_features_to_csv(df_features, csv_output)
    if save_to_db:
        save_features_to_db(df_features, db_name=db_name, table_name=table_name_features)
    
    logging.info("Feature engineering pipeline complete.")

def main():
    try:
        feature_engineering(
            source="db",
            db_name="my_trading_data.db",
            table_name_processed="ohlcv_processed",
            table_name_features="ohlcv_features",
            csv_input="BTC_USDT_15m_processed.csv",
            csv_output="BTC_USDT_15m_features.csv",
            save_to_csv=False,  # Set to False if you want to save only to DB
            save_to_db=True,
            add_extra_lags=True
        )
    except KeyboardInterrupt:
        logging.info("Feature engineering interrupted by user.")

if __name__ == "__main__":
    main()
