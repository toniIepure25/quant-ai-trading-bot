import os
import glob
import logging
import time
import sqlite3
import numpy as np
import pandas as pd
import datetime
import pytest
from unittest.mock import patch, MagicMock
from modules.paper_trading_simulation import PaperTradingSimulation

@pytest.fixture
def setup_database():
    # Setup a temporary SQLite database for testing
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE ohlcv_latent_features (date TEXT, feature1 REAL, feature2 REAL)''')
    cursor.execute('''INSERT INTO ohlcv_latent_features (date, feature1, feature2) VALUES (?, ?, ?)''', 
                   (datetime.datetime.now().isoformat(), 0.5, 1.5))
    conn.commit()
    yield conn
    conn.close()

@pytest.fixture
def setup_logging():
    logging.basicConfig(level=logging.INFO)

def test_load_historical_data(setup_database, setup_logging):
    simulation = PaperTradingSimulation(database_path=':memory:')
    data = simulation.load_historical_data()
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'date' in data.columns
    assert 'feature1' in data.columns
    assert 'feature2' in data.columns

@patch('modules.paper_trading_simulation.PaperTradingSimulation.load_historical_data')
@patch('modules.paper_trading_simulation.PaperTradingSimulation.load_ensemble_features')
def test_run_simulation(mock_load_historical_data, mock_load_ensemble_features, setup_database, setup_logging):
    mock_load_historical_data.return_value = pd.DataFrame({
        'date': [datetime.datetime.now().isoformat()],
        'feature1': [0.5],
        'feature2': [1.5]
    })
    mock_load_ensemble_features.return_value = np.array([[0.6, 0.4]])
    
    simulation = PaperTradingSimulation(database_path=':memory:')
    results = simulation.run_simulation()
    
    assert isinstance(results, dict)
    assert 'performance_metrics' in results
    assert 'summary' in results
    assert results['performance_metrics']['total_return'] >= 0

@patch('modules.paper_trading_simulation.PaperTradingSimulation.execute_trade')
def test_trade_execution(mock_execute_trade, setup_database, setup_logging):
    simulation = PaperTradingSimulation(database_path=':memory:')
    simulation.virtual_portfolio = {'cash': 10000, 'positions': {}}
    
    trade_signal = {'action': 'buy', 'amount': 100, 'price': 10}
    simulation.execute_trade(trade_signal)
    
    mock_execute_trade.assert_called_once_with(trade_signal)
    assert simulation.virtual_portfolio['cash'] == 9900
    assert 'AAPL' in simulation.virtual_portfolio['positions']

def test_logging_performance_metrics(setup_database, setup_logging):
    simulation = PaperTradingSimulation(database_path=':memory:')
    simulation.log_performance_metrics({'total_return': 0.1, 'sharpe_ratio': 1.5})
    
    # Check if logging occurred correctly
    with patch('logging.info') as mock_logging:
        simulation.log_performance_metrics({'total_return': 0.2, 'sharpe_ratio': 1.8})
        mock_logging.assert_called_with('Performance Metrics: %s', {'total_return': 0.2, 'sharpe_ratio': 1.8})

def test_edge_case_empty_data(setup_database, setup_logging):
    simulation = PaperTradingSimulation(database_path=':memory:')
    
    with patch('modules.paper_trading_simulation.PaperTradingSimulation.load_historical_data', return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No historical data available"):
            simulation.run_simulation()