import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from modules.modeling.supervised_model import SupervisedModel

@pytest.fixture(scope='module')
def test_db():
    # Setup a test database
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE ohlcv_latent_features (
            id INTEGER PRIMARY KEY,
            close REAL,
            log_return REAL
        )
    ''')
    # Insert test data
    cursor.executemany('''
        INSERT INTO ohlcv_latent_features (close, log_return) VALUES (?, ?)
    ''', [
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
    model = SupervisedModel(database_path=':memory:', table='ohlcv_latent_features')
    return model

def test_load_data(supervised_model):
    data = supervised_model.load_data()
    assert len(data) == 5
    assert 'close' in data.columns
    assert 'log_return' in data.columns

def test_prepare_target(supervised_model):
    data = supervised_model.load_data()
    target = supervised_model.prepare_target(data)
    assert len(target) == 5
    assert all(x in [0, 1] for x in target)

def test_split_data(supervised_model):
    data = supervised_model.load_data()
    train_data, test_data = supervised_model.split_data(data)
    assert len(train_data) + len(test_data) == len(data)
    assert len(train_data) > 0
    assert len(test_data) > 0

@patch('modules.modeling.supervised_model.XGBClassifier')
@patch('modules.modeling.supervised_model.LGBMClassifier')
@patch('modules.modeling.supervised_model.MLPClassifier')
def test_train_models(mock_mlp, mock_lgbm, mock_xgb, supervised_model):
    mock_xgb.return_value.fit = MagicMock()
    mock_lgbm.return_value.fit = MagicMock()
    mock_mlp.return_value.fit = MagicMock()

    data = supervised_model.load_data()
    target = supervised_model.prepare_target(data)
    supervised_model.train_models(data, target)

    assert mock_xgb.return_value.fit.called
    assert mock_lgbm.return_value.fit.called
    assert mock_mlp.return_value.fit.called

def test_compute_dynamic_weights(supervised_model):
    mock_losses = [0.1, 0.2, 0.15]
    mock_uncertainties = [0.05, 0.1, 0.07]
    weights = supervised_model.compute_dynamic_weights(mock_losses, mock_uncertainties)
    assert len(weights) == 3
    assert all(0 <= weight <= 1 for weight in weights)

def test_calibrate_model(supervised_model):
    mock_model = MagicMock()
    supervised_model.calibrate_model(mock_model)
    assert mock_model.temperature_scaling.called

def test_bayesian_optimization(supervised_model):
    mock_study = MagicMock()
    supervised_model.optimize_hyperparameters(mock_study)
    assert mock_study.optimize.called

def test_invalid_data_handling(supervised_model):
    with pytest.raises(ValueError):
        supervised_model.prepare_target(None)

def test_database_connection_error():
    with pytest.raises(sqlite3.OperationalError):
        SupervisedModel(database_path='invalid_path', table='ohlcv_latent_features')