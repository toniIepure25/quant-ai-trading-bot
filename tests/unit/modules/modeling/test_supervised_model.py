import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from modules.modeling.supervised_model import SupervisedModel

@pytest.fixture(scope='module')
def test_db():
    # Setup a test database connection
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE ohlcv_latent_features (
                        id INTEGER PRIMARY KEY,
                        close REAL,
                        log_return REAL)''')
    # Insert sample data
    cursor.executemany('INSERT INTO ohlcv_latent_features (close, log_return) VALUES (?, ?)', [
        (100.0, 0.01),
        (101.0, 0.02),
        (99.0, -0.01),
        (98.0, -0.02),
        (102.0, 0.03),
    ])
    connection.commit()
    yield connection
    connection.close()

@pytest.fixture
def supervised_model(test_db):
    model = SupervisedModel(db_path=':memory:', table='ohlcv_latent_features')
    return model

def test_load_data(supervised_model):
    data = supervised_model.load_data()
    assert len(data) == 5
    assert 'close' in data.columns
    assert 'log_return' in data.columns

def test_prepare_target(supervised_model):
    supervised_model.load_data()
    target = supervised_model.prepare_target()
    assert len(target) == 5
    assert all(x in [0, 1] for x in target)

def test_split_data(supervised_model):
    supervised_model.load_data()
    train_data, test_data = supervised_model.split_data()
    assert len(train_data) + len(test_data) == 5
    assert len(train_data) > 0
    assert len(test_data) > 0

@patch('modules.modeling.supervised_model.XGBClassifier')
@patch('modules.modeling.supervised_model.LGBMClassifier')
@patch('modules.modeling.supervised_model.MLPClassifier')
def test_train_models(mock_mlp, mock_lgbm, mock_xgb, supervised_model):
    mock_xgb.return_value = MagicMock()
    mock_lgbm.return_value = MagicMock()
    mock_mlp.return_value = MagicMock()
    
    supervised_model.load_data()
    supervised_model.train_models()
    
    mock_xgb.assert_called_once()
    mock_lgbm.assert_called_once()
    mock_mlp.assert_called_once()

def test_calibrate_model(supervised_model):
    supervised_model.load_data()
    supervised_model.train_models()
    calibration_result = supervised_model.calibrate_model()
    assert calibration_result is not None

def test_dynamic_weight_computation(supervised_model):
    supervised_model.load_data()
    supervised_model.train_models()
    weights = supervised_model.compute_dynamic_weights()
    assert len(weights) == 3  # Assuming three models
    assert all(isinstance(w, float) for w in weights)

def test_bayesian_optimization(supervised_model):
    supervised_model.load_data()
    optimization_result = supervised_model.optimize_hyperparameters()
    assert optimization_result is not None
    assert 'best_params' in optimization_result

def test_model_evaluation(supervised_model):
    supervised_model.load_data()
    supervised_model.train_models()
    evaluation_metrics = supervised_model.evaluate_model()
    assert 'accuracy' in evaluation_metrics
    assert evaluation_metrics['accuracy'] >= 0.5  # Assuming a baseline accuracy

def test_invalid_data_handling(supervised_model):
    # Test with invalid data
    with pytest.raises(ValueError):
        supervised_model.load_data(invalid=True)  # Assuming this raises an error

def test_data_validation(supervised_model):
    valid_data = supervised_model.validate_data()
    assert valid_data is True

    # Test with invalid data
    invalid_data = supervised_model.validate_data(invalid=True)
    assert invalid_data is False