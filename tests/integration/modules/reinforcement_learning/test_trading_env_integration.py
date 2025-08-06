import pytest
import sqlite3
from modules.reinforcement_learning.trading_env import TradingEnv

@pytest.fixture(scope="module")
def setup_database():
    # Setup a test SQLite database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create a table for market data
    cursor.execute('''
        CREATE TABLE market_data (
            timestamp INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    
    # Insert sample data
    sample_data = [
        (1, 100.0, 110.0, 90.0, 105.0, 1000.0),
        (2, 105.0, 115.0, 95.0, 110.0, 1500.0),
        (3, 110.0, 120.0, 100.0, 115.0, 2000.0),
    ]
    cursor.executemany('INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?)', sample_data)
    conn.commit()
    
    yield conn  # This will be the database connection used in tests
    
    # Teardown
    conn.close()

@pytest.fixture
def trading_env(setup_database):
    # Create an instance of the TradingEnv with the test database
    env = TradingEnv(database_path=':memory:')
    return env

def test_initialization(trading_env):
    assert trading_env is not None
    assert trading_env.action_space is not None
    assert trading_env.observation_space is not None

def test_reset(trading_env):
    state = trading_env.reset()
    assert state is not None
    assert len(state) > 0  # Ensure the state is not empty

def test_step(trading_env):
    trading_env.reset()
    action = trading_env.action_space.sample()  # Sample a random action
    state, reward, done, info = trading_env.step(action)
    
    assert state is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_reward_function(trading_env):
    trading_env.reset()
    action = trading_env.action_space.sample()
    _, reward, _, _ = trading_env.step(action)
    
    # Check if the reward is within expected bounds (this may need to be adjusted based on the actual reward function)
    assert reward <= 0  # Assuming the reward function penalizes excessive allocations

def test_database_integration(trading_env):
    trading_env.reset()
    action = trading_env.action_space.sample()
    state, reward, done, info = trading_env.step(action)
    
    # Verify that the environment interacts correctly with the database
    assert trading_env.current_step < len(trading_env.data)  # Ensure we haven't gone past the data
    assert trading_env.current_step >= 0  # Ensure current step is valid

def test_transaction_costs(trading_env):
    trading_env.reset()
    initial_balance = trading_env.balance
    action = trading_env.action_space.sample()
    _, reward, _, _ = trading_env.step(action)
    
    # Check if the balance reflects transaction costs
    assert trading_env.balance < initial_balance  # Assuming transaction costs reduce balance

def test_slippage_effect(trading_env):
    trading_env.reset()
    action = trading_env.action_space.sample()
    _, reward, _, _ = trading_env.step(action)
    
    # Check if slippage is accounted for in the reward
    assert reward < 0  # Assuming slippage negatively impacts reward

def test_edge_case_large_allocation(trading_env):
    trading_env.reset()
    action = trading_env.action_space.high  # Attempting a large allocation
    _, reward, _, _ = trading_env.step(action)
    
    # Check if the reward function penalizes excessive allocations
    assert reward < 0  # Expecting a penalty for large allocation

def test_edge_case_no_data(trading_env):
    trading_env.data = []  # Simulate no data scenario
    with pytest.raises(IndexError):  # Assuming an IndexError is raised when stepping with no data
        trading_env.step(0)  # Attempt to step with no data available