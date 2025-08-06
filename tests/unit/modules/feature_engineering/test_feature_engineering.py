import pytest
from unittest.mock import patch, MagicMock
from modules.feature_engineering.feature_engineering import FeatureEngineering

@pytest.fixture
def feature_engineering():
    return FeatureEngineering()

def test_load_data_csv(feature_engineering):
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = MagicMock()
        data = feature_engineering.load_data('data/processed/market/test_data.csv')
        mock_read_csv.assert_called_once_with('data/processed/market/test_data.csv')
        assert data is not None

def test_load_data_sqlite(feature_engineering):
    with patch('sqlite3.connect') as mock_connect:
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value.fetchall.return_value = [(1, 2, 3)]
        data = feature_engineering.load_data('data/processed/market/test_data.db', db=True)
        mock_connect.assert_called_once_with('data/processed/market/test_data.db')
        assert data is not None

def test_compute_technical_indicators(feature_engineering):
    sample_data = {
        'close': [1, 2, 3, 4, 5],
        'volume': [10, 20, 30, 40, 50]
    }
    indicators = feature_engineering.compute_technical_indicators(sample_data)
    assert 'SMA' in indicators
    assert 'EMA' in indicators
    assert 'RSI' in indicators

def test_compute_rolling_statistics(feature_engineering):
    sample_log_returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    stats = feature_engineering.compute_rolling_statistics(sample_log_returns)
    assert 'rolling_skewness' in stats
    assert 'rolling_kurtosis' in stats

def test_handle_empty_data(feature_engineering):
    with pytest.raises(ValueError):
        feature_engineering.compute_technical_indicators({})

def test_feature_engineering_full_pipeline(feature_engineering):
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = MagicMock()
        mock_read_csv.return_value.shape = (5, 2)
        data = feature_engineering.load_data('data/processed/market/test_data.csv')
        indicators = feature_engineering.compute_technical_indicators(data)
        stats = feature_engineering.compute_rolling_statistics(data['log_returns'])
        assert indicators is not None
        assert stats is not None

def test_invalid_database_path(feature_engineering):
    with pytest.raises(FileNotFoundError):
        feature_engineering.load_data('invalid/path/to/database.db', db=True)

def test_feature_engineering_with_mocked_data(feature_engineering):
    sample_data = {
        'close': [1, 2, 3, 4, 5],
        'volume': [10, 20, 30, 40, 50]
    }
    with patch.object(feature_engineering, 'compute_technical_indicators', return_value={'SMA': 3}) as mock_compute:
        indicators = feature_engineering.compute_technical_indicators(sample_data)
        mock_compute.assert_called_once_with(sample_data)
        assert indicators['SMA'] == 3

def test_feature_engineering_edge_case(feature_engineering):
    sample_data = {
        'close': [0, 0, 0, 0, 0],
        'volume': [0, 0, 0, 0, 0]
    }
    indicators = feature_engineering.compute_technical_indicators(sample_data)
    assert indicators['SMA'] == 0  # Assuming SMA of zeros is zero

def test_feature_engineering_performance(feature_engineering):
    import time
    sample_data = {
        'close': [1, 2, 3, 4, 5] * 1000,
        'volume': [10, 20, 30, 40, 50] * 1000
    }
    start_time = time.time()
    feature_engineering.compute_technical_indicators(sample_data)
    duration = time.time() - start_time
    assert duration < 1  # Ensure it runs within 1 second for large data
