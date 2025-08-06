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
    cursor.execute('CREATE TABLE ohlcv_latent (date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)')
    # Insert mock data
    cursor.execute('INSERT INTO ohlcv_latent VALUES (?, ?, ?, ?, ?, ?)', 
                   (datetime.datetime.now().isoformat(), 100.0, 105.0, 95.0, 102.0, 1000.0))
    conn.commit()
    yield conn
    conn.close()

@pytest.fixture
def trading_simulation(setup_database):
    # Create an instance of the PaperTradingSimulation with the mock database
    simulation = PaperTradingSimulation(database_connection=setup_database)
    return simulation

def test_load_historical_data(trading_simulation):
    # Test loading historical market data
    data = trading_simulation.load_historical_data('path/to/mock_data.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_load_latent_features(trading_simulation):
    # Test loading ensemble latent features from the database
    features = trading_simulation.load_latent_features()
    assert isinstance(features, pd.DataFrame)
    assert not features.empty

def test_generate_fused_signal(trading_simulation):
    # Mock the ensemble and RL agent methods
    with patch('modules.modeling.supervised_model.get_ensemble_predictions', return_value=np.array([0.5])) as mock_ensemble, \
         patch('modules.train_rl_agent.get_rl_signal', return_value=np.array([0.3])) as mock_rl:
        
        fused_signal = trading_simulation.generate_fused_signal()
        assert isinstance(fused_signal, np.ndarray)
        assert len(fused_signal) == 1
        assert fused_signal[0] == pytest.approx(0.4, rel=1e-2)  # Example calculation

def test_optimal_allocation(trading_simulation):
    # Test optimal allocation calculation
    allocation = trading_simulation.calculate_optimal_allocation(10000, 0.1)
    assert allocation > 0
    assert allocation <= 10000

def test_simulate_trade_execution(trading_simulation):
    # Test trade execution simulation
    initial_portfolio_value = trading_simulation.portfolio_value
    trading_simulation.simulate_trade_execution(1000, 'buy')
    assert trading_simulation.portfolio_value > initial_portfolio_value

def test_logging_performance_metrics(trading_simulation, caplog):
    # Test logging of performance metrics
    with caplog.at_level(logging.INFO):
        trading_simulation.log_performance_metrics()
    assert "Performance metrics logged" in caplog.text

def test_summary_of_simulated_returns(trading_simulation):
    # Test summary of simulated returns
    trading_simulation.simulate_trade_execution(1000, 'buy')
    summary = trading_simulation.get_summary()
    assert 'total_return' in summary
    assert summary['total_return'] >= 0  # Assuming no losses in this mock scenario

def test_integration_end_to_end(trading_simulation):
    # Test the end-to-end integration of the simulation
    trading_simulation.run_simulation()
    assert trading_simulation.portfolio_value > 0  # Ensure the portfolio has value after simulation

@pytest.mark.parametrize("trade_type, expected_value", [
    ('buy', 1),
    ('sell', -1),
])
def test_trade_execution_edge_cases(trading_simulation, trade_type, expected_value):
    # Test edge cases for trade execution
    initial_portfolio_value = trading_simulation.portfolio_value
    trading_simulation.simulate_trade_execution(1000, trade_type)
    assert (trading_simulation.portfolio_value - initial_portfolio_value) == expected_value * 1000

@pytest.mark.parametrize("invalid_trade_type", [
    'hold',
    'invalid',
])
def test_trade_execution_invalid_type(trading_simulation, invalid_trade_type):
    # Test invalid trade execution types
    with pytest.raises(ValueError):
        trading_simulation.simulate_trade_execution(1000, invalid_trade_type)