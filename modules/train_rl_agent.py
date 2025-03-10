#!/usr/bin/env python3
"""
Advanced RL Training Script for Trading Environment

Steps:
1. Loads advanced feature data from a local SQLite database using load_rl_data_from_db().
2. Splits data into training and evaluation sets.
3. Creates parallel SubprocVecEnv environments with custom TensorboardMonitor wrappers.
4. Trains a PPO agent with advanced logging:
   - TensorBoard,
   - Model checkpoints,
   - Evaluation callback.
5. Tests the trained model over multiple episodes and logs performance metrics.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Environment and custom wrappers
from modules.reinforcement_learning.trading_env import AdvancedTradingEnv, load_rl_data_from_db
from modules.reinforcement_learning.custom_wrappers import (
    TensorboardMonitor,
    make_tensorboard_env,
    TensorboardCallback
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def create_parallel_envs(data, n_envs=4, env_kwargs=None, tensorboard_log=None):
    """
    Create parallel environments with custom TensorboardMonitor.
    """
    if env_kwargs is None:
        env_kwargs = {}

    env_fns = []
    for i in range(n_envs):
        env_fns.append(
            make_tensorboard_env(
                env_class=AdvancedTradingEnv,
                data=data,
                tensorboard_log=f"{tensorboard_log}/env_{i}" if tensorboard_log else None,
                **env_kwargs
            )
        )

    vec_env = SubprocVecEnv(env_fns, start_method='fork')
    vec_env = VecMonitor(vec_env)  # record stats
    return vec_env


def split_data(data, train_ratio=0.8):
    """
    Split data into training and evaluation sets.
    """
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    eval_data = data.iloc[split_idx:].reset_index(drop=True)
    return train_data, eval_data


def train_agent(data, total_timesteps=100_000, n_envs=4, tensorboard_log="./ppo_tensorboard/"):
    """
    Trains a PPO agent on the advanced trading environment with TensorBoard logging.
    """
    os.makedirs(tensorboard_log, exist_ok=True)

    # Split data
    train_data, eval_data = split_data(data)

    # Common environment parameters
    env_kwargs = {
        "initial_cash": 10_000.0,
        "transaction_cost_pct": 0.001,
        "max_steps": 2000,
        "slippage_pct": 0.0005,
        "max_leverage": 5.0,
        "drawdown_penalty": True,
        "dynamic_threshold": True,
        "threshold_window": 10,
        "threshold_scale": 0.2,
        "static_threshold": 0.2,
    }

    # Create training/eval env
    env_train = create_parallel_envs(
        train_data,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        tensorboard_log=f"{tensorboard_log}/train"
    )
    env_eval = create_parallel_envs(
        eval_data,
        n_envs=1,
        env_kwargs=env_kwargs,
        tensorboard_log=f"{tensorboard_log}/eval"
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=f"{tensorboard_log}/best_model/",
        log_path=f"{tensorboard_log}/eval_logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{tensorboard_log}/checkpoints/",
        name_prefix='ppo_trading'
    )

    # Custom TensorBoard callback
    tensorboard_callback = TensorboardCallback()

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        device="cpu",  # or "cuda" if GPU is available
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=64,
        tensorboard_log=tensorboard_log,
    )

    logging.info(f"Starting PPO training for {total_timesteps} timesteps with {n_envs} parallel envs.")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, tensorboard_callback]
    )
    logging.info("Training complete.")

    env_train.close()
    env_eval.close()

    return model


def test_agent(model, data, n_episodes=5):
    """
    Test the trained agent, logging results.
    """
    env_kwargs = {
        "initial_cash": 10_000.0,
        "transaction_cost_pct": 0.001,
        "max_steps": len(data),
        "slippage_pct": 0.0005,
        "max_leverage": 5.0,
        "drawdown_penalty": True,
        "dynamic_threshold": True,
        "threshold_window": 10,
        "threshold_scale": 0.2,
        "static_threshold": 0.2,
    }

    # Single-environment VecEnv
    test_env = create_parallel_envs(
        data,
        n_envs=1,
        env_kwargs=env_kwargs,
        tensorboard_log="./ppo_tensorboard/test"
    )

    rewards = []
    for ep in range(n_episodes):
        reset_result = test_env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        total_reward = 0.0
        done = False
        step_counter = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = test_env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
                terminated = done
                truncated = False
            else:
                obs, reward, terminated, truncated, info = result
                done = terminated[0] or truncated[0]
            reward_val = reward[0] if np.ndim(reward) > 0 else reward
            total_reward += reward_val
            step_counter += 1
            
            # Log progress every 500 steps
            if step_counter % 500 == 0:
                logging.info(f"Episode {ep+1}: Step {step_counter}, Cumulative Reward: {total_reward:.2f}")
                
        rewards.append(total_reward)
        logging.info(f"[Test Episode {ep+1}] Total Reward: {total_reward:.2f}")

    test_env.close()
    avg_reward = np.mean(rewards)
    logging.info(f"Average Reward over {n_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def main():
    # Load data from DB
    data = load_rl_data_from_db(
        db_name="data/processed/my_trading_data.db",
        table_name="ohlcv_features"
    )
    logging.info(f"Data loaded. Shape={data.shape}")

    # Train
    model = train_agent(data, total_timesteps=100_000, n_envs=4)

    # Save final model
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"ppo_trading_agent_{timestamp}.zip"
    model.save(model_path)
    logging.info(f"Model saved as {model_path}")

    # Test
    test_agent(model, data, n_episodes=5)


if __name__ == "__main__":
    main()
