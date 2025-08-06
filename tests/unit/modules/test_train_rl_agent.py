import os
import pytest
from unittest.mock import patch, MagicMock
from modules.train_rl_agent import load_rl_data_from_db, train_rl_agent

@pytest.fixture
def mock_data():
    return {
        'features': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'labels': [0, 1]
    }

@pytest.fixture
def mock_env():
    env = MagicMock()
    env.reset.return_value = None
    return env

def test_load_rl_data_from_db(mocker):
    mocker.patch('modules.train_rl_agent.pd.read_sql', return_value=pd.DataFrame({
        'feature1': [0.1, 0.2],
        'feature2': [0.3, 0.4],
        'label': [0, 1]
    }))
    
    data = load_rl_data_from_db('mock_db_path')
    
    assert data['features'] == [[0.1, 0.3], [0.2, 0.4]]
    assert data['labels'] == [0, 1]

def test_train_rl_agent_success(mock_data, mock_env):
    with patch('modules.train_rl_agent.SubprocVecEnv') as mock_subproc_vec_env, \
         patch('modules.train_rl_agent.PPO') as mock_ppo, \
         patch('modules.train_rl_agent.EvalCallback') as mock_eval_callback, \
         patch('modules.train_rl_agent.CheckpointCallback') as mock_checkpoint_callback:
        
        mock_subproc_vec_env.return_value = mock_env
        mock_ppo.return_value = MagicMock()
        
        train_rl_agent(mock_data)
        
        mock_subproc_vec_env.assert_called_once()
        mock_ppo.assert_called_once()
        mock_ppo.return_value.learn.assert_called_once()
        mock_eval_callback.assert_called_once()
        mock_checkpoint_callback.assert_called_once()

def test_train_rl_agent_failure(mock_data):
    with patch('modules.train_rl_agent.PPO') as mock_ppo:
        mock_ppo.side_effect = Exception("Training failed")
        
        with pytest.raises(Exception, match="Training failed"):
            train_rl_agent(mock_data)

def test_train_rl_agent_logging(mock_data, caplog):
    with patch('modules.train_rl_agent.SubprocVecEnv'), \
         patch('modules.train_rl_agent.PPO'):
        
        train_rl_agent(mock_data)
        
        assert "Training started" in caplog.text
        assert "Training completed" in caplog.text

def test_train_rl_agent_evaluation(mock_data):
    with patch('modules.train_rl_agent.EvalCallback') as mock_eval_callback:
        train_rl_agent(mock_data)
        
        mock_eval_callback.assert_called_once()