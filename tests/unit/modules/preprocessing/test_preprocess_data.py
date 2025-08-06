import os
import sqlite3
import logging
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from modules.preprocessing.preprocess_data import preprocess_data

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def sample_data():
    """Fixture to provide sample OHLCV data for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H'),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [104, 105, 106, 107, 108],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })

@pytest.fixture
def db_connection():
    """Fixture to create a temporary SQLite database for testing."""
    conn = sqlite3.connect(':memory:')
    yield conn
    conn.close()

def test_data_validation(sample_data):
    """Test data validation functionality."""
    result = preprocess_data.validate_data(sample_data)
    assert result is True

def test_outlier_removal_z_score(sample_data):
    """Test outlier removal using Z-score method."""
    # Introduce an outlier
    sample_data.loc[2, 'close'] = 200
    cleaned_data = preprocess_data.remove_outliers(sample_data, method='z-score')
    assert cleaned_data['close'].iloc[2] != 200  # Outlier should be removed

def test_outlier_removal_iqr(sample_data):
    """Test outlier removal using IQR method."""
    # Introduce an outlier
    sample_data.loc[2, 'close'] = 200
    cleaned_data = preprocess_data.remove_outliers(sample_data, method='IQR')
    assert cleaned_data['close'].iloc[2] != 200  # Outlier should be removed

def test_normalization_minmax(sample_data):
    """Test min-max normalization."""
    normalized_data = preprocess_data.normalize_data(sample_data, method='minmax')
    assert normalized_data['open'].min() == 0
    assert normalized_data['open'].max() == 1

def test_normalization_standard(sample_data):
    """Test standard normalization."""
    normalized_data = preprocess_data.normalize_data(sample_data, method='standard')
    assert np.isclose(normalized_data['open'].mean(), 0, atol=1e-2)
    assert np.isclose(normalized_data['open'].std(), 1, atol=1e-2)

def test_cleaning_duplicates(sample_data):
    """Test cleaning of duplicate entries."""
    sample_data = sample_data.append(sample_data.iloc[0])  # Introduce a duplicate
    cleaned_data = preprocess_data.clean_data(sample_data)
    assert len(cleaned_data) == len(sample_data) - 1  # One duplicate should be removed

def test_cleaning_missing_values(sample_data):
    """Test cleaning of missing values."""
    sample_data.loc[2, 'close'] = np.nan  # Introduce a missing value
    cleaned_data = preprocess_data.clean_data(sample_data)
    assert cleaned_data['close'].isnull().sum() == 0  # No missing values should remain

def test_save_processed_data(db_connection, sample_data):
    """Test saving processed data to the database."""
    preprocess_data.save_to_db(sample_data, db_connection, table_name='ohlcv_data')
    query_result = pd.read_sql_query("SELECT * FROM ohlcv_data", db_connection)
    assert len(query_result) == len(sample_data)  # All rows should be saved

def test_invalid_data_format():
    """Test handling of invalid data format."""
    with pytest.raises(ValueError):
        preprocess_data.validate_data("invalid_data")  # Should raise ValueError

def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    empty_data = pd.DataFrame()
    result = preprocess_data.validate_data(empty_data)
    assert result is False  # Validation should fail for empty DataFrame

def test_database_connection_error():
    """Test handling of database connection errors."""
    with patch('sqlite3.connect', side_effect=sqlite3.OperationalError):
        with pytest.raises(sqlite3.OperationalError):
            preprocess_data.save_to_db(pd.DataFrame(), 'invalid_connection', 'ohlcv_data')  # Should raise OperationalError