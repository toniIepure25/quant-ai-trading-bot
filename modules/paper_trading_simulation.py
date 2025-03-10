#!/usr/bin/env python3
"""
Paper Trading Simulation Module

This module simulates real-time (paper) trading using historical market data.
It:
  1. Loads historical market data from a CSV (simulating a real-time feed).
  2. Loads ensemble latent features from a database.
  3. At each time step, uses the unified trading system to generate a fused signal:
       - Ensemble (meta-model) predictions from the supervised ensemble.
       - RL agent's trading signal.
  4. Computes the optimal allocation using a Kelly Criterion–inspired formula.
  5. Simulates trade execution by updating a virtual portfolio.
  6. Logs performance metrics and outputs a summary of simulated returns.

Usage:
  $ python paper_trading_simulation.py
"""

import os
import glob
import logging
import time
import sqlite3
import numpy as np
import pandas as pd
import datetime

from stable_baselines3 import PPO
from tensorflow.keras.models import load_model

# Import unified trading functions from our integration module.
# (We assume that these functions are defined in trading_system_integration.py.)
from trading_system_integration import (
    load_ensemble_features,
    load_ensemble_meta_model,
    load_rl_agent,
    prepare_features,
    get_ensemble_signal,
    get_rl_signal,
    fuse_signals,
    compute_optimal_allocation,
    simulate_trade
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Additional Helper Function
# -----------------------------
def prepare_meta_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the three meta feature columns for the meta–model.
    If the expected columns ['xgb_pred', 'lgb_pred', 'deep_pred'] are not present,
    fall back to using ['latent_7', 'latent_8', 'latent_9'].
    """
    required_cols = ['xgb_pred', 'lgb_pred', 'deep_pred']
    fallback_cols = ['latent_7', 'latent_8', 'latent_9']
    
    if all(col in df.columns for col in required_cols):
        meta_df = df[required_cols]
        logger.info(f"Using required meta columns: {required_cols}")
    elif all(col in df.columns for col in fallback_cols):
        meta_df = df[fallback_cols]
        logger.info(f"Required meta columns not found. Using fallback columns: {fallback_cols}")
    else:
        raise ValueError("Neither the required meta columns nor the fallback columns are present in the data.")
    
    # Convert to float32 numpy array
    return meta_df.astype("float32").values

# -----------------------------
# Simulation Configuration
# -----------------------------
INITIAL_CASH = 10000.0
SIMULATION_INTERVAL = 0.1  # seconds between simulated "ticks" (set shorter for demo)
# CSV file with market (RL) features – make sure this file exists and has a 'close' column.
DATA_SOURCE = os.path.join("data", "processed", "market", "BTC_USDT_15m_features.csv")

# -----------------------------
# Paper Trading Simulator
# -----------------------------
def run_paper_trading_simulation() -> (pd.DataFrame, pd.DataFrame):
    # --- Load simulated market data from CSV (RL features) ---
    if not os.path.exists(DATA_SOURCE):
        raise FileNotFoundError(f"Market data file not found: {DATA_SOURCE}")
    market_data = pd.read_csv(DATA_SOURCE)
    market_data["timestamp"] = pd.to_datetime(market_data["timestamp"], utc=True)
    market_data = market_data.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded market data with {market_data.shape[0]} rows from CSV.")

    # Prepare RL features from market data (all numeric columns)
    X_rl = prepare_features(market_data)
    logger.info(f"RL features shape: {X_rl.shape}")

    # --- Load ensemble latent features from DB (for meta-model) ---
    ensemble_df = load_ensemble_features(db_name="my_trading_data.db", table_name="ohlcv_latent_features")
    # Use our dedicated function to extract only the meta features (3 columns)
    X_meta = prepare_meta_features(ensemble_df)
    logger.info(f"Meta features shape: {X_meta.shape}")

    # --- Load pre-trained models ---
    meta_model = load_ensemble_meta_model(model_path="models/ensemble/meta_model_ensemble.keras")
    rl_agent = load_rl_agent()  # This searches the trading_agent folder for the latest agent.

    # --- Initialize Portfolio ---
    cash = INITIAL_CASH
    position = 0.0  # Fraction of portfolio currently invested (can be negative for shorts)
    portfolio_value = INITIAL_CASH
    trade_history = []
    performance = []

    # --- Simulation Loop ---
    # We'll iterate over the minimum number of ticks available among the data sources.
    n_ticks = min(market_data.shape[0], X_rl.shape[0], X_meta.shape[0])
    for idx in range(n_ticks):
        row = market_data.iloc[idx]
        current_price = row["close"]

        # Get the corresponding features for this tick:
        # For the meta-model, we use the prepared meta features (shape: (1, 3))
        current_meta = X_meta[idx].reshape(1, -1)
        # For the RL agent, we use the RL features (from the CSV)
        current_rl = X_rl[idx].reshape(1, -1)

        # Obtain signals from the models
        ensemble_signal = get_ensemble_signal(meta_model, current_meta)
        rl_signal = get_rl_signal(rl_agent, current_rl)
        fused_signal = fuse_signals(ensemble_signal, rl_signal, alpha=0.6)
        optimal_allocation = compute_optimal_allocation(fused_signal, expected_return=0.02, variance=0.05)

        # Simulate trade execution: compute order value to adjust portfolio allocation
        order_value = simulate_trade(position, optimal_allocation, portfolio_value)
        # Update portfolio:
        position += order_value / portfolio_value  # Update position (fractional change)
        cash -= order_value                      # Deduct order value from cash
        portfolio_value = cash + position * current_price  # Recalculate portfolio value

        # Log performance metrics for this tick
        performance.append({
            "timestamp": row["timestamp"],
            "current_price": current_price,
            "position": position,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "optimal_allocation": optimal_allocation,
            "fused_signal": fused_signal,
            "ensemble_signal": float(ensemble_signal.mean()),
            "rl_signal": rl_signal
        })
        trade_history.append({
            "timestamp": row["timestamp"],
            "order_value": order_value,
            "new_position": position
        })

        # Optional: simulate real-time interval (for demonstration)
        time.sleep(SIMULATION_INTERVAL)

        if idx % 50 == 0:
            logger.info(f"Tick {idx}: Portfolio Value = {portfolio_value:.2f}")

    # Convert logs to DataFrames for analysis and optionally save to CSV files
    performance_df = pd.DataFrame(performance)
    trades_df = pd.DataFrame(trade_history)
    performance_df.to_csv("paper_trading_performance.csv", index=False)
    trades_df.to_csv("paper_trading_history.csv", index=False)
    logger.info("Paper trading simulation complete.")
    logger.info(f"Final Portfolio Value: {portfolio_value:.2f}")

    return performance_df, trades_df

if __name__ == "__main__":
    run_paper_trading_simulation()
