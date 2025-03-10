#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Optimization for the Hybrid Ensemble

This script wraps the hybrid ensemble training pipeline with Optuna,
optimizing hyperparameters such as:
  - n_splits for time-series cross-validation
  - lambda_risk for the meta-learner's custom loss function

The objective is to maximize the AUC on the validation set.
By default, Optuna minimizes the objective, so we return the negative AUC.

Usage:
  !python optuna_ensemble_optimization.py
"""

import optuna
import logging
import os
import pandas as pd
from advanced_hybrid_supervised_ensemble_improved import load_features_from_db, prepare_data, train_hybrid_ensemble

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Load the data once outside the objective (assuming the DB is available)
def load_data():
    # Update these parameters if needed
    df = load_features_from_db(db_name="my_trading_data.db", table_name="ohlcv_latent_features")
    X, y = prepare_data(df)
    return X, y

X, y = load_data()

def objective(trial):
    # Suggest hyperparameters to optimize
    n_splits = trial.suggest_int("n_splits", 3, 7)
    lambda_risk = trial.suggest_float("lambda_risk", 0.05, 0.2)
    
    # You could also add hyperparameters for base model training here if you update the ensemble training function
    # e.g., xgb_rounds = trial.suggest_int("xgb_rounds", 100, 300)

    # Train the hybrid ensemble with the current hyperparameters
    ensemble_results = train_hybrid_ensemble(X, y, n_splits=n_splits, lambda_risk=lambda_risk)
    
    # Our evaluation metric is AUC (we want it to be as high as possible)
    auc = ensemble_results['validation_auc']
    
    # Since Optuna minimizes the objective, we return negative AUC
    return -auc

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    logging.info("Best trial:")
    trial = study.best_trial
    logging.info(f"  Value: {-trial.value:.4f} AUC")  # Multiply by -1 to show positive AUC
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")
    
    # Optionally, you can save the study for later analysis:
    study.trials_dataframe().to_csv("optuna_study_results.csv", index=False)
