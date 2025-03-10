#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for Advanced Feature Extraction with Database Integration

This module:
  1. Loads engineered feature data from a local SQLite database 
     (default: data/processed/my_trading_data.db, table='ohlcv_features').
  2. Normalizes the input features (saving mean and std for later use).
  3. Trains a Variational Autoencoder (VAE) on the normalized feature set 
     to extract a robust, low-dimensional latent representation.
  4. Saves the trained encoder and normalization parameters.
  5. Uses the trained encoder (with the saved normalization parameters) to extract latent features.
  6. Appends the latent features as new columns to the original features.
  7. Saves the combined DataFrame back into the SQLite database 
     (e.g. table 'ohlcv_latent_features').

Usage:
  !python modules/feature_engineering/train_vae.py
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Create a directory for caching if needed
from joblib import Memory
memory = Memory(location='./cache', verbose=0)

# --- Sampling layer for the VAE ---
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- VAE Model Definition ---
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, optimizer):
        super(VAE, self).compile()
        self.optimizer = optimizer
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            # Clip log variance to prevent extreme values.
            z_log_var = tf.clip_by_value(z_log_var, -5.0, 5.0)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(keras.losses.mse(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        clipped_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        self.optimizer.apply_gradients(zip(clipped_grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# --- Encoder and Decoder Builders ---
def build_encoder(input_dim, latent_dim, intermediate_dims=[128, 64], l2_factor=1e-4):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for dim in intermediate_dims:
        x = layers.Dense(dim, activation="relu", kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l2_factor))(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary(print_fn=logging.info)
    return encoder

def build_decoder(latent_dim, output_dim, intermediate_dims=[64, 128], l2_factor=1e-4):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    for dim in intermediate_dims:
        x = layers.Dense(dim, activation="relu", kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l2_factor))(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    decoder.summary(print_fn=logging.info)
    return decoder

# --- Database I/O Functions ---
def load_features_from_db(db_name="my_trading_data.db", table_name="ohlcv_features") -> pd.DataFrame:
    """Load engineered features from the specified SQLite database table."""
    db_path = os.path.join("data", "processed", db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Processed DB not found at: {db_path}")
    logging.info(f"Loading engineered features from DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    logging.info(f"Loaded {df.shape[0]} rows from engineered features table.")
    return df

def save_combined_features_to_db(df: pd.DataFrame, db_name="my_trading_data.db", table_name="ohlcv_latent_features"):
    """Save the combined DataFrame (engineered features + latent features) into the database."""
    db_path = os.path.join("data", "processed", db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    logging.info(f"Saving combined features to DB: {db_path}, table={table_name}")
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logging.info("Combined feature data successfully stored in the DB.")

# --- VAE Training and Latent Feature Extraction ---
def train_vae_on_db(db_name="my_trading_data.db", table_name="ohlcv_features",
                     latent_dim=10, epochs=50, batch_size=32,
                     intermediate_dims_encoder=[128, 64], intermediate_dims_decoder=[64, 128],
                     encoder_save_dir="models/vae", encoder_filename="vae_encoder.h5",
                     norm_params_filename="vae_norm.npz"):
    """
    Load engineered features from the DB, normalize them (and save the stats),
    train a VAE on the normalized features, and save the encoder and normalization parameters.
    """
    df = load_features_from_db(db_name, table_name)
    # Remove non-numeric columns (e.g., timestamp) before training.
    if "timestamp" in df.columns:
        df_train = df.drop(columns=["timestamp"])
    else:
        df_train = df.copy()
    X = df_train.values.astype("float32")
    # Normalize features: standardization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std
    logging.info(f"Feature data stats after normalization: min={np.min(X_norm):.4f}, max={np.max(X_norm):.4f}, mean={np.mean(X_norm):.4f}")
    
    # Save normalization parameters for later use during inference.
    os.makedirs(encoder_save_dir, exist_ok=True)
    norm_params_path = os.path.join(encoder_save_dir, norm_params_filename)
    np.savez(norm_params_path, X_mean=X_mean, X_std=X_std)
    logging.info(f"Normalization parameters saved to {norm_params_path}")
    
    input_dim = X_norm.shape[1]
    encoder = build_encoder(input_dim, latent_dim, intermediate_dims_encoder)
    decoder = build_decoder(latent_dim, input_dim, intermediate_dims_decoder)
    vae = VAE(encoder, decoder)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    vae.compile(optimizer=optimizer)
    logging.info("Starting VAE training...")
    vae.fit(X_norm, epochs=epochs, batch_size=batch_size, shuffle=True)
    logging.info("VAE training complete.")
    
    # Save the encoder to the designated directory
    encoder_save_path = os.path.join(encoder_save_dir, encoder_filename)
    encoder.save(encoder_save_path)
    logging.info(f"Encoder saved to {encoder_save_path}")
    return vae, encoder, df

def extract_and_append_latent_features(df: pd.DataFrame, encoder_path="models/vae/vae_encoder.h5",
                                         norm_params_path="models/vae/vae_norm.npz", latent_prefix="latent_") -> pd.DataFrame:
    """
    Load the trained encoder and normalization parameters, apply the same normalization to the input,
    extract latent features, and append them to the original DataFrame.
    """
    # Load normalization parameters
    norm_data = np.load(norm_params_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    
    encoder = keras.models.load_model(encoder_path, custom_objects={"Sampling": Sampling})
    
    # Exclude non-numeric columns for latent extraction (but keep them for later concatenation)
    feature_df = df.copy()
    non_numeric = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_df = feature_df.drop(columns=non_numeric) if non_numeric else feature_df
    X = numeric_df.values.astype("float32")
    # Apply the same normalization used during training
    X_norm = (X - X_mean) / X_std
    
    # Predict latent features
    _, _, z = encoder.predict(X_norm)
    # Replace any Inf values in z with NaN (and later fill or drop)
    z[np.isinf(z)] = np.nan
    latent_df = pd.DataFrame(z, columns=[f"{latent_prefix}{i}" for i in range(z.shape[1])], index=numeric_df.index)
    
    # Join the latent features with the original DataFrame using the index
    combined_df = feature_df.join(latent_df)
    # Option: Instead of dropping rows with NaN, you can fill them (here we fill with column medians)
    combined_df.fillna(combined_df.median(numeric_only=True), inplace=True)
    logging.info(f"Combined features shape after processing: {combined_df.shape}")
    return combined_df

# --- Main Pipeline ---
def main():
    """
    1. Load engineered features from the DB (table: 'ohlcv_features').
    2. Train a VAE on these features, saving the encoder and normalization parameters.
    3. Extract latent features using the trained encoder and saved normalization stats.
    4. Append latent features to the original features.
    5. Save the combined DataFrame to a new table in the DB (table: 'ohlcv_latent_features').
    """
    try:
        vae_model, trained_encoder, df_features = train_vae_on_db(
            db_name="my_trading_data.db",
            table_name="ohlcv_features",
            latent_dim=10,
            epochs=50,
            batch_size=32,
            intermediate_dims_encoder=[128, 64],
            intermediate_dims_decoder=[64, 128],
            encoder_save_dir="models/vae",
            encoder_filename="vae_encoder.h5",
            norm_params_filename="vae_norm.npz"
        )
        combined_df = extract_and_append_latent_features(
            df_features, 
            encoder_path="models/vae/vae_encoder.h5", 
            norm_params_path="models/vae/vae_norm.npz", 
            latent_prefix="latent_"
        )
        save_combined_features_to_db(combined_df, db_name="my_trading_data.db", table_name="ohlcv_latent_features")
    except Exception as e:
        logging.error("Error in VAE pipeline: " + str(e))
        raise

if __name__ == "__main__":
    main()
