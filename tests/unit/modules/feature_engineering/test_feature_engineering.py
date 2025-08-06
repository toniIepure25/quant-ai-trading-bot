import pytest
from unittest.mock import patch, MagicMock
from modules.feature_engineering.feature_engineering import FeatureEngineering

@pytest.fixture
def feature_engineering():
    return FeatureEngineering()

def test_load_data_from_csv(feature_engineering):
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = MagicMock()
        data = feature_engineering.load_data_from_csv('data/processed/market/test.csv')
        mock_read_csv.assert_called_once_with('data/processed/market/test.csv')
        assert data is not None

def test_load_data_from_db(feature_engineering):
    with patch('sqlite3.connect') as mock_connect:
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        data = feature_engineering.load_data_from_db('data/processed/my_trading_data.db')
        mock_connect.assert_called_once_with('data/processed/my_trading_data.db')
        assert data is not None

def test_compute_log_returns(feature_engineering):
    sample_data = [100, 105, 102, 108]
    expected_returns = [0.0500, -0.0286, 0.0588]
    returns = feature_engineering.compute_log_returns(sample_data)
    assert returns == pytest.approx(expected_returns, rel=1e-2)

def test_compute_sma(feature_engineering):
    sample_data = [1, 2, 3, 4, 5]
    expected_sma = [None, None, 2.0, 3.0, 4.0]
    sma = feature_engineering.compute_sma(sample_data, window=3)
    assert sma == expected_sma

def test_compute_ema(feature_engineering):
    sample_data = [1, 2, 3, 4, 5]
    expected_ema = [1.0, 1.5, 2.0, 3.0, 4.0]
    ema = feature_engineering.compute_ema(sample_data, span=3)
    assert ema == pytest.approx(expected_ema, rel=1e-2)

def test_compute_rsi(feature_engineering):
    sample_data = [44, 45, 46, 47, 48, 49, 50]
    expected_rsi = 100.0  # Example expected value
    rsi = feature_engineering.compute_rsi(sample_data)
    assert rsi == pytest.approx(expected_rsi, rel=1e-2)

def test_handle_empty_data(feature_engineering):
    with pytest.raises(ValueError, match="Data cannot be empty"):
        feature_engineering.compute_log_returns([])

def test_feature_engineering_pipeline(feature_engineering):
    with patch.object(feature_engineering, 'load_data_from_csv', return_value=[100, 105, 102, 108]) as mock_load:
        with patch.object(feature_engineering, 'compute_log_returns', return_value=[0.0500, -0.0286, 0.0588]) as mock_compute:
            feature_engineering.run_pipeline('data/processed/market/test.csv')
            mock_load.assert_called_once()
            mock_compute.assert_called_once()

def test_invalid_file_path(feature_engineering):
    with pytest.raises(FileNotFoundError):
        feature_engineering.load_data_from_csv('invalid/path.csv')

def test_compute_bollinger_bands(feature_engineering):
    sample_data = [1, 2, 3, 4, 5]
    expected_bands = (2.5, 1.5, 3.5)  # Example expected values for (middle, lower, upper)
    bands = feature_engineering.compute_bollinger_bands(sample_data, window=3, num_std=1)
    assert bands == pytest.approx(expected_bands, rel=1e-2)