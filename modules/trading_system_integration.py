#!/usr/bin/env python3
"""
Unified Trading System Integration Module

This module integrates:
  - The risk-sensitive supervised ensemble predictions from your database.
  - The RL agent’s trading signal.
  - Signal fusion via weighted averaging.
  - Portfolio optimization via a Kelly Criterion–inspired approach.
  - A simulated order execution (for backtesting/scenario analysis).
  - Real-time monitoring via logging.

Usage:
  $ python trading_system_integration.py
"""

import os
import glob
import logging
import sqlite3
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorflow.keras.models import load_model
from typing import Tuple

# Setup logging with a custom logger for clarity.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ----------------------------
# 1. Data & Model Loading
# ----------------------------
def load_ensemble_features(db_name: str = "my_trading_data.db", table_name: str = "ohlcv_latent_features") -> pd.DataFrame:
    """
    Load combined engineered and latent features from the SQLite database.
    """
    db_path = os.path.join("data", "processed", db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    logger.info(f"Loading ensemble features from DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from ensemble features.")
    return df

def prepare_meta_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare meta features. We require that one of the following sets of columns is present:
      - Preferred: ['xgb_pred', 'lgb_pred', 'deep_pred']
      - If not, fallback to using a set of latent columns (e.g. ['latent_7', 'latent_8', 'latent_9'])
    """
    required = ['xgb_pred', 'lgb_pred', 'deep_pred']
    if all(col in df.columns for col in required):
        meta_cols = required
    else:
        fallback = ['latent_7', 'latent_8', 'latent_9']
        if all(col in df.columns for col in fallback):
            logger.warning(f"Required meta columns {required} not found. Fallback to using columns: {fallback}")
            meta_cols = fallback
        else:
            raise ValueError(f"Neither required meta columns {required} nor fallback {fallback} found in data.")
    meta_features = df[meta_cols].values.astype("float32")
    logger.info(f"Meta features shape: {meta_features.shape}")
    return meta_features

def load_rl_features(db_name: str = "my_trading_data.db", table_name: str = "ohlcv_features") -> np.ndarray:
    """
    Load features for the RL agent from the specified DB table.
    """
    db_path = os.path.join("data", "processed", db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    logger.info(f"Loading RL features from DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Drop non-numeric columns (like timestamp)
    numeric_df = df.select_dtypes(include=[np.number])
    rl_features = numeric_df.values.astype("float32")
    logger.info(f"RL features shape: {rl_features.shape}")
    return rl_features

def load_ensemble_meta_model(model_path: str = "models/ensemble/meta_model_ensemble.keras"):
    """
    Load the pre-trained meta-learner model in the native Keras format.
    """
    logger.info(f"Loading ensemble meta-model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model file not found: {model_path}")
    return load_model(model_path, compile=False)

def load_rl_agent(model_path: str = None):
    """
    Load the pre-trained PPO RL agent.
    If no explicit model_path is provided, this function searches the
    "models/trading_agent/" directory for files matching the pattern
    "ppo_trading_agent_*.zip" and loads the most recently modified file.
    """
    folder = "models/trading_agent"
    pattern = os.path.join(folder, "ppo_trading_agent_*.zip")
    model_files = glob.glob(pattern) if model_path is None else [model_path]
    
    if not model_files:
        raise FileNotFoundError(f"No RL agent model file found in {folder} with pattern {pattern}")
    
    latest_model = max(model_files, key=os.path.getmtime)
    logger.info(f"Loading RL agent from {latest_model}")
    return PPO.load(latest_model)

# ----------------------------
# 2. Signal Fusion & Portfolio Optimization
# ----------------------------
def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare features from a DataFrame by dropping non-numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.values.astype("float32")

def get_ensemble_signal(meta_model, X_meta: np.ndarray) -> np.ndarray:
    """
    Obtain the ensemble prediction signal from the meta-learner.
    """
    logits = meta_model.predict(X_meta).flatten()
    # Apply temperature scaling (T=1.0 for simplicity; adjust as needed)
    T = 1.0
    signal = 1 / (1 + np.exp(-logits / T))
    logger.info(f"Ensemble signal computed (mean: {np.mean(signal):.4f}).")
    return signal

def create_dummy_rl_env(observation: np.ndarray, expected_shape: Tuple[int, ...]) -> gym.Env:
    """
    Create a simple dummy environment for RL agent prediction.
    This environment returns an observation padded to the expected shape.
    """
    # Pad the observation if needed
    if observation.shape != expected_shape:
        padded = np.zeros(expected_shape, dtype=observation.dtype)
        padded[:observation.shape[0]] = observation
        observation = padded

    class DummyEnv(gym.Env):
        def __init__(self, obs: np.ndarray):
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            # Create a dummy continuous action space with one dimension.
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.obs = obs
        def reset(self, seed=None, options=None):
            return self.obs, {}  # Return observation and an empty info dict.
        def step(self, action):
            return self.obs, 0.0, True, False, {}
    return DummyEnv(observation)

def get_rl_signal(rl_agent, X_rl: np.ndarray) -> float:
    """
    Use the RL agent to obtain a trading signal.
    We use the last row of RL features as the current observation.
    """
    current_obs = X_rl[-1]
    expected_shape = rl_agent.observation_space.shape
    # Create dummy env with observation padded to the expected shape.
    env = DummyVecEnv([lambda: create_dummy_rl_env(current_obs, expected_shape)])
    
    # Reset the environment. Depending on the version, reset may return a tuple or a single value.
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result

    action, _ = rl_agent.predict(obs, deterministic=True)
    signal = float(action[0])  # Assuming a scalar action.
    logger.info(f"RL signal computed: {signal:.4f}")
    return signal

def fuse_signals(ensemble_signal: np.ndarray, rl_signal: float, alpha: float = 0.6) -> float:
    """
    Fuse ensemble and RL signals using a weighted average.
    Alpha controls the weight for the ensemble; (1 - alpha) for RL.
    """
    fused = alpha * np.mean(ensemble_signal) + (1 - alpha) * rl_signal
    logger.info(f"Fused signal: {fused:.4f}")
    return fused

def compute_optimal_allocation(signal: float, expected_return: float = 0.02, variance: float = 0.05) -> float:
    """
    Compute optimal allocation using a Kelly Criterion–inspired formula.
    """
    kelly_fraction = expected_return / (variance + 1e-8)
    allocation = np.clip(signal * kelly_fraction, -1, 1)
    logger.info(f"Optimal allocation (Kelly-based): {allocation:.4f}")
    return allocation

# ----------------------------
# 3. Trade Execution Simulation
# ----------------------------
def simulate_trade(current_position: float, optimal_allocation: float, portfolio_value: float) -> float:
    """
    Simulate order execution: compute the trade order value needed to move from the current position
    to the target (optimal) allocation.
    """
    target_value = optimal_allocation * portfolio_value
    trade_value = target_value - current_position * portfolio_value
    logger.info(f"Simulated trade order value: {trade_value:.2f}")
    return trade_value

# ----------------------------
# 4. Unified Trading System Integration
# ----------------------------
def main():
    try:
        # Load ensemble features (for meta-model predictions) and RL features from the DB.
        df_latent = load_ensemble_features(db_name="my_trading_data.db", table_name="ohlcv_latent_features")
        X_meta = prepare_meta_features(df_latent)
        X_rl = load_rl_features(db_name="my_trading_data.db", table_name="ohlcv_features")
        
        # Load pre-trained models.
        meta_model = load_ensemble_meta_model(model_path="models/ensemble/meta_model_ensemble.keras")
        rl_agent = load_rl_agent()  # Will load the latest model from the trading_agent folder.
        
        # Obtain signals from the ensemble meta-model and RL agent.
        ensemble_signal = get_ensemble_signal(meta_model, X_meta)
        rl_signal = get_rl_signal(rl_agent, X_rl)
        
        # Fuse the signals to get a final trading signal.
        fused_signal = fuse_signals(ensemble_signal, rl_signal, alpha=0.6)
        
        # Compute optimal allocation using a Kelly-based approach.
        optimal_allocation = compute_optimal_allocation(fused_signal, expected_return=0.02, variance=0.05)
        
        # Simulate trade execution using example portfolio values.
        current_position = 0.03   # e.g., 3% current allocation.
        portfolio_value = 1000.0  # Example portfolio value.
        trade_value = simulate_trade(current_position, optimal_allocation, portfolio_value)
        
        logger.info("Unified trading system integration complete.")
    except Exception as e:
        logger.error("Error in unified trading system pipeline: " + str(e))
        raise

if __name__ == "__main__":
    main()
