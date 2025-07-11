# pytest configuration for unit tests with mocking
import pytest
from unittest.mock import patch, Mock
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# pytest configuration for unit tests with mocking
import pytest
from unittest.mock import patch, Mock
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture(autouse=True)
def mock_heavy_imports():
    """Mock heavy imports to speed up unit tests."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('transformers.set_seed'):
            with patch('diffusers.DiffusionPipeline.from_pretrained'):
                with patch('diffusers.FluxPipeline.from_pretrained'):
                    with patch('diffusers.ControlNetModel.from_pretrained'):
                        yield

@pytest.fixture
def mock_torch_device():
    """Mock torch.device for tests."""
    with patch('torch.device') as mock_device:
        mock_device.return_value = Mock()
        yield mock_device

@pytest.fixture
def mock_logger():
    """Mock logger for tests."""
    with patch('loguru.logger') as mock_log:
        yield mock_log

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for tests."""
    env_vars = {
        'BUCKET_NAME': 'test-bucket',
        'BUCKET_PATH': 'test-path',
        'STORAGE_PROVIDER': 'r2',
        'HF_HOME': '/tmp/test_cache',
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    from PIL import Image
    return Image.new('RGB', (512, 512), color='red')

@pytest.fixture
def mock_custom_pipeline():
    """Mock CustomPipeline for tests."""
    with patch('stable_diffusion_server.custom_pipeline.CustomPipeline') as mock_pipeline:
        mock_instance = Mock()
        mock_pipeline.return_value = mock_instance
        mock_instance.generate.return_value = b"fake_image_bytes"
        yield mock_instance
