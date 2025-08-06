```python
import pytest
import os
import tempfile
import pandas as pd
from unittest.mock import MagicMock

@pytest.fixture(scope='module')
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def mock_dataframe():
    """Create a mock DataFrame for testing."""
    data = {
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_preprocess_function(mocker):
    """Mock a preprocessing function."""
    return mocker.patch('modules.preprocessing.preprocess_data.preprocess_function', return_value=None)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'param1': 'value1',
        'param2': 'value2',
        'param3': 10
    }

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Common setup and teardown for tests."""
    # Setup code here
    yield
    # Teardown code here

def create_test_file(temp_dir, filename, content):
    """Utility function to create a test file."""
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, 'w') as f:
        f.write(content)
    return file_path

def assert_dataframe_equal(df1, df2):
    """Utility function to assert two DataFrames are equal."""
    pd.testing.assert_frame_equal(df1, df2)

@pytest.fixture
def mock_file(temp_dir):
    """Create a mock file for testing."""
    file_path = create_test_file(temp_dir, 'test_file.csv', 'column1,column2\n1,a\n2,b\n3,c')
    return file_path
```