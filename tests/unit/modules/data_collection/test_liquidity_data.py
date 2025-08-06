import os
import pytest
import requests
import pandas as pd
from unittest.mock import patch, MagicMock
from modules.data_collection.liquidity_data import fetch_liquidity_data, save_liquidity_to_csv

class TestLiquidityData:

    @patch('requests.get')
    def test_fetch_liquidity_data_success(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Data": {
                "Exchanges": [
                    {"NAME": "Exchange1", "VOLUME24H": 1000},
                    {"NAME": "Exchange2", "VOLUME24H": 2000}
                ]
            }
        }
        mock_get.return_value = mock_response
        
        # Act
        df = fetch_liquidity_data()
        
        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "NAME" in df.columns
        assert df["NAME"].tolist() == ["Exchange1", "Exchange2"]

    @patch('requests.get')
    def test_fetch_liquidity_data_failure(self, mock_get):
        # Arrange
        mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error: Not Found for url")
        
        # Act & Assert
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_liquidity_data()

    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_liquidity_to_csv(self, mock_to_csv, mock_makedirs):
        # Arrange
        df = pd.DataFrame({"NAME": ["Exchange1", "Exchange2"], "VOLUME24H": [1000, 2000]})
        filename = "test_liquidity_data.csv"
        
        # Act
        save_liquidity_to_csv(df, filename)
        
        # Assert
        mock_makedirs.assert_called_once_with(os.path.join("data", "unprocessed", "liquidity"), exist_ok=True)
        mock_to_csv.assert_called_once_with(os.path.join("data", "unprocessed", "liquidity", filename), index=False)

    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_liquidity_to_csv_default_filename(self, mock_to_csv, mock_makedirs):
        # Arrange
        df = pd.DataFrame({"NAME": ["Exchange1", "Exchange2"], "VOLUME24H": [1000, 2000]})
        
        # Act
        save_liquidity_to_csv(df)
        
        # Assert
        mock_makedirs.assert_called_once_with(os.path.join("data", "unprocessed", "liquidity"), exist_ok=True)
        mock_to_csv.assert_called_once_with(os.path.join("data", "unprocessed", "liquidity", "liquidity_data.csv"), index=False)