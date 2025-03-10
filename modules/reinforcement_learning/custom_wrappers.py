#!/usr/bin/env python3
"""
Custom TensorBoard Monitor and Callback wrappers for Advanced Trading Environment.

This module contains:
  - TensorboardMonitor: A gym wrapper that adapts to gymnasium's 5-value return,
    logs detailed trading metrics to TensorBoard, and tracks episode returns.
  - make_tensorboard_env: A helper function to create an environment wrapped with TensorboardMonitor.
  - TensorboardCallback: A callback for logging additional training metrics to TensorBoard.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from gym import Env
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf
import logging

class TensorboardMonitor(Monitor):
    """
    Custom Monitor wrapper that:
      1. Handles gymnasium's 5-value return from step().
      2. Logs detailed trading metrics to TensorBoard.
      3. Properly resets and tracks episode metrics.
    """
    
    def __init__(
        self,
        env: gym.Env,
        log_dir: Optional[str] = None,
        allow_early_resets: bool = True,
        info_keywords: Tuple[str, ...] = (),
        tensorboard_log: Optional[str] = None,
    ):
        super().__init__(env, log_dir, allow_early_resets, info_keywords)
        
        # Set up TensorBoard logging
        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log:
            os.makedirs(self.tensorboard_log, exist_ok=True)
            self.writer = tf.summary.create_file_writer(self.tensorboard_log)
        else:
            self.writer = None
        
        # Trackers for trading metrics and episode statistics
        self.episode_counter = 0
        self.step_counter = 0
        self.portfolio_values = []
        self.actions_taken = []
        self.positions = []
        self.cash = []
        
        # New variables to track episode rewards and lengths
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_episode_return = 0.0
        
        # Backward compatibility for gym vs. gymnasium interfaces:
        # If env.step has fewer than 5 arguments, assume Gym-style.
        self._is_gym_style = hasattr(env, 'step') and len(env.step.__code__.co_varnames) < 5
        
    def reset(self, **kwargs):
        """
        Handle environment reset for both gym and gymnasium interfaces.
        """
        # Reset trackers for the new episode
        self.portfolio_values = []
        self.actions_taken = []
        self.positions = []
        self.cash = []
        self.step_counter = 0
        self.current_episode_return = 0.0
        
        result = self.env.reset(**kwargs)
        
        if isinstance(result, tuple):
            # Gymnasium style: (observation, info)
            obs, info = result
            logging.info(
                f"Environment reset: current_step={self.step_counter}, "
                f"end_step={getattr(self.env, 'max_steps', 'N/A')}, "
                f"initial portfolio value={getattr(self.env, 'portfolio_value', 0.0):.2f}"
            )
            if 'portfolio_value' in info:
                self.portfolio_values.append(info['portfolio_value'])
            if 'position' in info:
                self.positions.append(info['position'])
            if 'cash' in info:
                self.cash.append(info['cash'])
            return obs, info
        else:
            # Gym style: observation only
            obs = result
            logging.info(
                f"Environment reset: current_step={self.step_counter}, "
                f"end_step={getattr(self.env, 'max_steps', 'N/A')}, "
                f"initial portfolio value={getattr(self.env, 'portfolio_value', 0.0):.2f}"
            )
            if hasattr(self.env, 'portfolio_value'):
                self.portfolio_values.append(self.env.portfolio_value)
            if hasattr(self.env, 'position'):
                self.positions.append(self.env.position)
            if hasattr(self.env, 'cash'):
                self.cash.append(self.env.cash)
            return obs
            
    def step(self, action):
        """
        Execute one environment step, record metrics, and format the return appropriately.
        """
        self.actions_taken.append(action)
        self.step_counter += 1
        
        result = self.env.step(action)
        
        # Handle Gym vs. Gymnasium interface
        if self._is_gym_style or len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated
        
        # Accumulate reward for this episode
        self.current_episode_return += reward
        
        # Record trading metrics if available
        if 'portfolio_value' in info:
            self.portfolio_values.append(info['portfolio_value'])
        elif hasattr(self.env, 'portfolio_value'):
            self.portfolio_values.append(self.env.portfolio_value)
            
        if 'position' in info:
            self.positions.append(info['position'])
        elif hasattr(self.env, 'position'):
            self.positions.append(self.env.position)
            
        if 'cash' in info:
            self.cash.append(info['cash'])
        elif hasattr(self.env, 'cash'):
            self.cash.append(self.env.cash)
        
        # If episode finished, log metrics and reset episode-specific counters
        if done or truncated:
            self.episode_counter += 1
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.step_counter)
            self._log_episode_metrics()
            # Reset for next episode
            self.current_episode_return = 0.0
            self.step_counter = 0
            self.portfolio_values = []
            self.actions_taken = []
            self.positions = []
            self.cash = []
        
        # Return in Gymnasium format (5 values) if applicable
        if self._is_gym_style or len(result) == 4:
            return obs, reward, done, truncated, info
        else:
            return obs, reward, terminated, truncated, info
    
    def _log_episode_metrics(self):
        """
        Log the episode's trading metrics to TensorBoard.
        """
        if self.writer is None or not self.episode_returns or not self.episode_lengths:
            return
        
        with self.writer.as_default():
            tf.summary.scalar('episode/reward', self.episode_returns[-1], step=self.episode_counter)
            tf.summary.scalar('episode/length', self.episode_lengths[-1], step=self.episode_counter)
            
            # Trading metrics: calculate returns percentage and maximum drawdown
            if self.portfolio_values:
                initial_value = self.portfolio_values[0]
                final_value = self.portfolio_values[-1]
                returns_pct = ((final_value / initial_value) - 1.0) * 100
                
                max_drawdown = 0
                peak = self.portfolio_values[0]
                for value in self.portfolio_values:
                    peak = max(peak, value)
                    drawdown = (peak - value) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
                
                tf.summary.scalar('trading/final_portfolio_value', final_value, step=self.episode_counter)
                tf.summary.scalar('trading/return_percentage', returns_pct, step=self.episode_counter)
                tf.summary.scalar('trading/max_drawdown', max_drawdown, step=self.episode_counter)
                
                # Log the portfolio value curve over the episode
                for i, value in enumerate(self.portfolio_values):
                    tf.summary.scalar('trading/portfolio_curve', value, step=i)
            
            # Log action distribution if discrete
            if self.actions_taken and isinstance(self.actions_taken[0], (int, np.integer)):
                action_counts = np.bincount(self.actions_taken)
                for action, count in enumerate(action_counts):
                    tf.summary.scalar(f'actions/action_{action}', count, step=self.episode_counter)
            
            self.writer.flush()
    
    def seed(self, seed=None):
        """Pass seed to the environment."""
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return []
    
    def close(self):
        """Close the environment and TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
        return self.env.close()


def make_tensorboard_env(env_class, data, tensorboard_log=None, **env_kwargs):
    """
    Create an environment wrapped with TensorboardMonitor.
    
    :param env_class: The trading environment class.
    :param data: Data to pass to the environment.
    :param tensorboard_log: Path for TensorBoard logs.
    :param env_kwargs: Additional environment keyword arguments.
    """
    def _init():
        env = env_class(data=data, **env_kwargs)
        wrapped_env = TensorboardMonitor(
            env,
            tensorboard_log=tensorboard_log
        )
        return wrapped_env
    return _init


class TensorboardCallback(BaseCallback):
    """
    Callback for logging additional training metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log additional metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'dump'):
                explained_var = self.model.logger.name_to_value.get('train/explained_variance', 0)
                policy_loss = self.model.logger.name_to_value.get('train/policy_loss', 0)
                value_loss = self.model.logger.name_to_value.get('train/value_loss', 0)
                
                self.logger.record('custom/explained_variance', explained_var)
                self.logger.record('custom/policy_loss', policy_loss)
                self.logger.record('custom/value_loss', value_loss)
                
        return True
