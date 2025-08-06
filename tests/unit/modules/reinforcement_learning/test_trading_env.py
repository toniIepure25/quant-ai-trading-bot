import pytest
from unittest.mock import patch, MagicMock
from modules.reinforcement_learning.trading_env import TradingEnv

@pytest.fixture
def trading_env():
    env = TradingEnv(database_path='data/processed/my_trading_data.db')
    yield env
    env.close()

def test_initialization(trading_env):
    assert trading_env is not None
    assert trading_env.action_space is not None
    assert trading_env.observation_space is not None

def test_reset(trading_env):
    state = trading_env.reset()
    assert state is not None
    assert len(state) == trading_env.observation_space.shape[0]

def test_step(trading_env):
    state = trading_env.reset()
    action = trading_env.action_space.sample()
    next_state, reward, done, info = trading_env.step(action)

    assert next_state is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_step_penalizes_excessive_allocations(trading_env):
    trading_env.current_allocation = 1.5  # Simulate excessive allocation
    action = trading_env.action_space.sample()
    next_state, reward, done, info = trading_env.step(action)

    assert reward < 0  # Expect a penalty for excessive allocation

def test_load_data_from_db(trading_env):
    with patch('modules.reinforcement_learning.trading_env.load_data_from_db') as mock_load:
        mock_load.return_value = MagicMock()
        trading_env.load_data_from_db()
        mock_load.assert_called_once()

def test_render(trading_env):
    trading_env.reset()
    trading_env.render()  # Ensure no exceptions are raised

def test_close(trading_env):
    trading_env.close()  # Ensure no exceptions are raised on close

def test_action_space(trading_env):
    assert trading_env.action_space.n > 0  # Ensure action space is valid

def test_observation_space(trading_env):
    assert trading_env.observation_space.shape[0] > 0  # Ensure observation space is valid

def test_reward_function(trading_env):
    action = trading_env.action_space.sample()
    trading_env.step(action)
    assert trading_env.reward_function() is not None  # Ensure reward function returns a value

def test_invalid_action(trading_env):
    with pytest.raises(ValueError):
        trading_env.step(-1)  # Test invalid action handling

def test_done_condition(trading_env):
    trading_env.reset()
    for _ in range(100):  # Simulate multiple steps
        action = trading_env.action_space.sample()
        _, _, done, _ = trading_env.step(action)
        if done:
            break
    assert done is True  # Ensure done condition is met

def test_transaction_costs(trading_env):
    initial_balance = trading_env.balance
    action = trading_env.action_space.sample()
    trading_env.step(action)
    assert trading_env.balance < initial_balance  # Ensure balance is reduced by transaction costs

def test_slippage(trading_env):
    initial_position = trading_env.current_position
    action = trading_env.action_space.sample()
    trading_env.step(action)
    assert trading_env.current_position != initial_position  # Ensure position changes due to slippage

def test_edge_case_empty_data(trading_env):
    with patch('modules.reinforcement_learning.trading_env.load_data_from_db', return_value=[]):
        with pytest.raises(ValueError):
            trading_env.reset()  # Ensure it raises an error on empty data

def test_edge_case_negative_allocation(trading_env):
    trading_env.current_allocation = -0.5  # Simulate negative allocation
    action = trading_env.action_space.sample()
    next_state, reward, done, info = trading_env.step(action)

    assert reward < 0  # Expect a penalty for negative allocation