#!/usr/bin/env python3
"""
Advanced Trading Environment for RL with Database Integration

This module implements a Gymnasium-compatible trading environment designed
for AI quant trading on heavy-tailed markets (e.g. BTC, BTC futures). It loads
preprocessed market data directly from a local SQLite database, and its reward
function is designed to be risk-sensitive by penalizing excessive allocations.
It also simulates realistic trading by incorporating transaction costs and slippage.

Key Features:
  - Loads data from a SQLite database (ensuring fast I/O and central data management).
  - Uses a continuous action space representing a target allocation (allowing for leveraged positions).
  - The observation space includes both market features (all columns except standard OHLCV) 
    and portfolio state (current position, normalized cash, normalized portfolio value, current price).
  - The reward function calculates the percentage change in portfolio value from one step to the next,
    then subtracts a risk penalty proportional to the absolute target allocation.
  - Compatible with advanced RL training pipelines (e.g., using PPO).

Usage:
  !python modules/reinforcement_learning/trading_env.py
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_rl_data_from_db(db_name="my_trading_data.db", table_name="ohlcv_features") -> pd.DataFrame:
    """
    Load processed market data from the specified SQLite database table.
    You can change table_name to "ohlcv_latent_features" if using latent features.
    """
    db_path = os.path.join("data", "processed", db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    logging.info(f"Loading market data from DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    logging.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from DB.")
    return df

class AdvancedTradingEnv(gym.Env):
    """
    A risk-sensitive trading environment for reinforcement learning.
    
    Action Space:
      - A continuous value in [-max_leverage, max_leverage] representing the target allocation
        (as a multiple of the current portfolio value relative to the asset price).
    
    Observation Space:
      - Consists of market features (all columns except standard OHLCV columns) concatenated with:
          • Current position (number of shares or fraction)
          • Normalized cash (cash divided by initial cash)
          • Normalized portfolio value (portfolio value divided by initial cash)
          • Current asset price
      
    Reward:
      - Calculated as the relative change in portfolio value from the previous step.
      - A risk penalty is subtracted proportional to the absolute action (allocation level).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_cash: float = 10000.0,
                 transaction_cost_pct: float = 0.001, slippage_pct: float = 0.0005,
                 max_steps: int = None, max_leverage: float = 5.0, risk_penalty_coef: float = 0.1):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_steps = max_steps if max_steps is not None else len(data)
        self.max_leverage = max_leverage
        self.risk_penalty_coef = risk_penalty_coef
        
        # Define action space: target allocation (can be negative for shorts)
        self.action_space = spaces.Box(low=-max_leverage, high=max_leverage, shape=(1,), dtype=np.float32)
        
        # Determine feature columns: exclude standard OHLCV columns
        standard_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        self.feature_columns = [col for col in data.columns if col not in standard_cols]
        # Observation: market features + 4 extra dimensions for portfolio state
        obs_dim = len(self.feature_columns) + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Initialize state variables
        self.reset()

    def seed(self, seed=None):
      np.random.seed(seed)
      return [seed]

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        # For price, we use the 'close' column from the data.
        self.current_price = self.data.loc[0, 'close']
        # Store previous portfolio value for incremental reward calculation.
        self.prev_portfolio_value = self.initial_cash
        return self._get_obs(), {}

    def _get_obs(self):
        # Extract market features from the current row
        features = self.data.loc[self.current_step, self.feature_columns].values.astype(np.float32)
        # Additional portfolio state information:
        extra = np.array([
            self.position, 
            self.cash / self.initial_cash, 
            self.portfolio_value / self.initial_cash, 
            self.current_price
        ], dtype=np.float32)
        return np.concatenate([features, extra])

    def step(self, action):
        # Clip action to allowed range
        action = float(np.clip(action, -self.max_leverage, self.max_leverage))
        # Determine target position: allocation * (portfolio_value / current_price)
        target_position = action * self.portfolio_value / self.current_price
        trade_size = target_position - self.position
        # Effective price incorporates slippage
        effective_price = self.current_price * (1 + np.sign(trade_size) * self.slippage_pct)
        transaction_cost = abs(trade_size * effective_price * self.transaction_cost_pct)
        # Update cash and position
        self.cash -= (trade_size * effective_price + transaction_cost)
        self.position = target_position
        # Save portfolio value from previous step for reward calculation
        self.prev_portfolio_value = self.portfolio_value
        # Advance step
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.current_step >= len(self.data)
        if not done:
            self.current_price = self.data.loc[self.current_step, 'close']
        # Update portfolio value: cash + position * current_price
        self.portfolio_value = self.cash + self.position * self.current_price
        # Compute reward: incremental return (percentage change) minus risk penalty
        if self.prev_portfolio_value > 0:
            step_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        else:
            step_return = 0.0
        risk_penalty = self.risk_penalty_coef * abs(action)
        reward = step_return - risk_penalty
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'current_price': self.current_price
        }
        return self._get_obs(), reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Price: {self.current_price:.2f}, "
              f"Position: {self.position:.4f}, Cash: {self.cash:.2f}, "
              f"Portfolio Value: {self.portfolio_value:.2f}")

    def close(self):
        pass  # Any required cleanup

# If run as main, demonstrate loading data from DB and resetting the env.
if __name__ == "__main__":
    try:
        # Load market data from database (adjust table name as needed, e.g., 'ohlcv_features' or 'ohlcv_latent_features')
        df_market = load_rl_data_from_db(db_name="my_trading_data.db", table_name="ohlcv_features")
        # Create the trading environment
        env = AdvancedTradingEnv(data=df_market, initial_cash=10000.0, risk_penalty_coef=0.1)
        obs, info = env.reset()
        print("Initial observation:", obs)
        # Take a random action as a test
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    except Exception as e:
        logging.error("Error in trading environment: " + str(e))
