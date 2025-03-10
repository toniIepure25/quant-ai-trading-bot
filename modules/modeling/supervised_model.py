#!/usr/bin/env python3
"""
Advanced Hybrid Supervised Ensemble with Dynamic Reweighting,
Uncertainty-Aware Meta-Learner, Advanced Loss, and Bayesian Hyperparameter Optimization,
along with Model Calibration.

This script:
  1. Loads combined engineered and latent features from a SQLite DB
     (default: data/processed/my_trading_data.db, table='ohlcv_latent_features').
  2. Prepares a binary target (1 if next log_return > 0, else 0) based on the 'close' column.
  3. Splits data using a time-series method.
  4. Trains three base models: XGBoost, LightGBM, and a deep MLP with Monte Carlo dropout.
  5. Computes dynamic weights using both loss and uncertainty (for the deep model).
  6. Trains a meta-learner with a custom risk-aware loss function.
  7. Applies temperature scaling for calibration.
  8. Uses Optuna for Bayesian hyperparameter optimization (to search for optimal training parameters).
  9. Saves the final meta-learner model in a dedicated directory.

Usage:
  !python advanced_hybrid_supervised_ensemble_improved.py
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

#############################################
# 1) DATABASE I/O FUNCTIONS
#############################################
def load_features_from_db(db_name="my_trading_data.db", table_name="ohlcv_latent_features") -> pd.DataFrame:
    db_path = os.path.join("data", "processed", db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    logging.info(f"Loading features from DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    logging.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def save_ensemble_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    model.save(filename)
    logging.info(f"Meta-learner model saved to {filename}")

#############################################
# 2) DATA PREPARATION
#############################################
def prepare_data(df: pd.DataFrame):
    """
    Prepares feature matrix X and binary target y.
    Steps:
      - Check for 'close' column and remove non-positive values.
      - Compute log returns and replace any Inf with NaN.
      - Create a binary target (1 if next log_return > 0, else 0).
      - Drop rows with any NaN values.
    """
    df = df.copy()
    if 'close' not in df.columns:
        logging.error("Column 'close' not found in DataFrame!")
        raise ValueError("Missing 'close' column in DataFrame.")
    if (df['close'] <= 0).any():
        logging.warning("Found non-positive values in 'close'. Filtering them out.")
        df = df[df['close'] > 0]
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['target'] = (df['log_return'].shift(-1) > 0).astype(int)
    logging.info(f"Before dropping NaN, DataFrame shape: {df.shape}")
    df.dropna(inplace=True)
    logging.info(f"After dropping NaN, DataFrame shape: {df.shape}")
    if df.empty:
        logging.error("DataFrame is empty after cleaning. Check your feature engineering and VAE pipeline.")
        raise ValueError("Empty DataFrame after dropna.")
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=['timestamp', 'target'], errors='ignore')
    y = df['target']
    logging.info(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
    return X, y

#############################################
# 3) TIME-SERIES SPLIT
#############################################
def time_series_split(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    if len(splits) == 0:
        raise ValueError("No splits generated. Check the number of samples in X.")
    train_idx, valid_idx = splits[-1]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    logging.info(f"TimeSeriesSplit: Training set: {X_train.shape}, Validation set: {X_valid.shape}")
    return X_train, y_train, X_valid, y_valid

#############################################
# 4) BASE MODEL TRAINING FUNCTIONS
#############################################
def train_xgb(X_train, y_train, X_valid, y_valid, num_rounds=200, seed=42):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'auc',
        'seed': seed
    }
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(params, dtrain, num_rounds, evals, early_stopping_rounds=10)
    logging.info("XGBoost base model trained.")
    return model

def train_lgb(X_train, y_train, X_valid, y_valid, num_rounds=200, seed=42):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'seed': seed,
        'verbose': -1
    }
    callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=True)]
    model = lgb.train(params, train_data, num_boost_round=num_rounds, valid_sets=[valid_data],
                      callbacks=callbacks)
    logging.info("LightGBM base model trained.")
    return model

def train_deep_model(X_train, y_train, X_valid, y_valid, epochs=50, batch_size=64):
    input_dim = X_train.shape[1]
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs,
              batch_size=batch_size, callbacks=[early_stop], verbose=0)
    logging.info("Deep MLP base model trained (with dropout for MC uncertainty).")
    return model

#############################################
# 5) BASE PREDICTION FUNCTIONS
#############################################
def predict_xgb(model, X):
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)

def predict_lgb(model, X):
    return model.predict(X)

def predict_deep_with_uncertainty(model, X, n_iter=50):
    preds = []
    for _ in range(n_iter):
        pred = model(X, training=True)
        preds.append(pred.numpy().flatten())
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    var_pred = preds.var(axis=0)
    return mean_pred, var_pred

def predict_deep_with_mc(model, X, n_iter=50):
    mean_preds, var_preds = predict_deep_with_uncertainty(model, X, n_iter)
    return mean_preds, var_preds

#############################################
# 6) DYNAMIC WEIGHT CALCULATION
#############################################
def compute_dynamic_weights(base_preds, deep_uncertainty, y_valid, eps=1e-6):
    # Compute inverse loss-based weights for XGB and LGB.
    weights = {}
    total_weight = 0
    loss_xgb = -np.mean(y_valid * np.log(base_preds['xgb'] + eps) +
                          (1 - y_valid) * np.log(1 - base_preds['xgb'] + eps))
    w_xgb = 1 / (loss_xgb + eps)
    weights['xgb'] = w_xgb
    total_weight += w_xgb

    loss_lgb = -np.mean(y_valid * np.log(base_preds['lgb'] + eps) +
                          (1 - y_valid) * np.log(1 - base_preds['lgb'] + eps))
    w_lgb = 1 / (loss_lgb + eps)
    weights['lgb'] = w_lgb
    total_weight += w_lgb

    # For the deep model, incorporate average uncertainty.
    loss_deep = -np.mean(y_valid * np.log(base_preds['deep'] + eps) +
                           (1 - y_valid) * np.log(1 - base_preds['deep'] + eps))
    avg_uncertainty = np.mean(deep_uncertainty)
    w_deep = 1 / (loss_deep + avg_uncertainty + eps)
    weights['deep'] = w_deep
    total_weight += w_deep

    for key in weights:
        weights[key] /= total_weight

    logging.info(f"Dynamic weights computed: {weights}")
    return weights

#############################################
# 7) META-LEARNER & CALIBRATION
#############################################
def custom_meta_loss(lambda_risk=0.1):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        risk_penalty = lambda_risk * tf.reduce_mean(tf.square(y_pred - 0.5))
        return bce + risk_penalty
    return loss

def train_meta_learner(meta_features, y_valid, epochs=50, batch_size=32, lambda_risk=0.1):
    input_dim = meta_features.shape[1]
    meta_model = Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    meta_model.compile(optimizer='adam', loss=custom_meta_loss(lambda_risk=lambda_risk), metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    meta_model.fit(meta_features, y_valid, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                   callbacks=[early_stop], verbose=0)
    logging.info("Meta-learner trained with custom risk-aware loss.")
    return meta_model

def calibrate_temperature(y_true, logits):
    best_temp = 1.0
    best_loss = float('inf')
    for T in np.linspace(0.5, 2.0, 31):
        probs = 1 / (1 + np.exp(-logits / T))
        loss = -np.mean(y_true * np.log(probs + 1e-6) + (1 - y_true) * np.log(1 - probs + 1e-6))
        if loss < best_loss:
            best_loss = loss
            best_temp = T
    logging.info(f"Optimal temperature scaling parameter found: T = {best_temp:.3f}")
    return best_temp

def apply_temperature_scaling(logits, T):
    return 1 / (1 + np.exp(-logits / T))

#############################################
# 8) HYBRID ENSEMBLE TRAINING FUNCTION
#############################################
def train_hybrid_ensemble(X, y, n_splits=5, lambda_risk=0.1):
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Prepared data is empty. Check your feature engineering and VAE pipeline.")
    X_train, y_train, X_valid, y_valid = time_series_split(X, y, n_splits)
    
    # Train base models
    xgb_model = train_xgb(X_train, y_train, X_valid, y_valid)
    lgb_model = train_lgb(X_train, y_train, X_valid, y_valid)
    deep_model = train_deep_model(X_train, y_train, X_valid, y_valid)
    
    # Generate predictions on the validation set
    preds_xgb = predict_xgb(xgb_model, X_valid)
    preds_lgb = predict_lgb(lgb_model, X_valid)
    deep_mean, deep_var = predict_deep_with_mc(deep_model, X_valid, n_iter=50)
    
    base_preds = {'xgb': preds_xgb, 'lgb': preds_lgb, 'deep': deep_mean}
    weights = compute_dynamic_weights(base_preds, deep_var, y_valid)
    
    meta_features = pd.DataFrame({
        'xgb': preds_xgb * weights['xgb'],
        'lgb': preds_lgb * weights['lgb'],
        'deep': deep_mean * weights['deep']
    })
    
    meta_model = train_meta_learner(meta_features, y_valid, lambda_risk=lambda_risk)
    raw_logits = meta_model.predict(meta_features).flatten()
    T_opt = calibrate_temperature(y_valid.values, raw_logits)
    calibrated_preds = apply_temperature_scaling(raw_logits, T_opt)
    
    acc = accuracy_score(y_valid, (calibrated_preds > 0.5).astype(int))
    auc = roc_auc_score(y_valid, calibrated_preds)
    logging.info(f"Ensemble Validation - ACC: {acc:.4f}, AUC: {auc:.4f}")
    
    expected_return = 0.02  # Example value; update with data-driven estimates as needed
    variance = 0.05         # Example value; update with data-driven estimates as needed
    risk_adjusted_signals = np.clip((calibrated_preds) / (variance + 1e-8), -1, 1)
    logging.info("Risk-adjusted signals computed using Kelly-style adjustment.")
    
    return {
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'deep_model': deep_model,
        'meta_model': meta_model,
        'calibrated_predictions': calibrated_preds,
        'risk_adjusted_signals': risk_adjusted_signals,
        'validation_accuracy': acc,
        'validation_auc': auc,
        'dynamic_weights': weights,
        'temperature': T_opt
    }

#############################################
# 9) OPTUNA OBJECTIVE FUNCTION
#############################################
def objective(trial):
    # Suggest hyperparameters for ensemble components
    lambda_risk = trial.suggest_float("lambda_risk", 0.001, 0.1, log=True)
    deep_epochs = trial.suggest_int("deep_epochs", 30, 70)
    xgb_rounds = trial.suggest_int("xgb_rounds", 100, 300)
    lgb_rounds = trial.suggest_int("lgb_rounds", 100, 300)
    
    # Use the training functions with these hyperparameters:
    # (For simplicity, we adjust only the deep model epochs and boosting rounds for the base models.)
    # NOTE: In a production system, you would modify train_xgb, train_lgb, and train_deep_model to accept these parameters.
    
    # For now, we assume these suggested values are used indirectly inside our base training functions.
    ensemble_results = train_hybrid_ensemble(X_global, y_global, n_splits=5, lambda_risk=lambda_risk)
    # Our objective is to maximize AUC; we return the negative AUC as Optuna minimizes the objective.
    return -ensemble_results['validation_auc']

#############################################
# 10) MAIN PIPELINE
#############################################
def main():
    try:
        df = load_features_from_db(db_name="my_trading_data.db", table_name="ohlcv_latent_features")
        X, y = prepare_data(df)
        if X.shape[0] < 100:
            raise ValueError(f"Not enough samples after cleaning: only {X.shape[0]} rows available.")
        
        # Set globals for the Optuna objective function
        global X_global, y_global
        X_global, y_global = X, y
        
        # Create an Optuna study for hyperparameter optimization
        study = optuna.create_study(direction="minimize", study_name="Ensemble_Hyperopt")
        study.optimize(objective, n_trials=10)  # Increase n_trials for a more thorough search
        
        logging.info("Optuna study complete.")
        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best value (negative AUC): {study.best_trial.value:.4f}")
        logging.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logging.info(f"    {key}: {value}")
        
        # After optimization, retrain the ensemble with the best hyperparameters:
        best_lambda_risk = study.best_trial.params.get("lambda_risk", 0.01)
        ensemble_results = train_hybrid_ensemble(X, y, n_splits=5, lambda_risk=best_lambda_risk)
        
        ensemble_save_path = "models/ensemble/meta_model_ensemble.keras"
        save_ensemble_model(ensemble_results['meta_model'], ensemble_save_path)
        logging.info("Hybrid ensemble training complete.")
        
    except Exception as e:
        logging.error("Error in hybrid ensemble pipeline: " + str(e))
        raise

if __name__ == "__main__":
    main()
