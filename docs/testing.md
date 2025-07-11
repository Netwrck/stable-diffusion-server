# ControlNet Unit Testing Documentation

## Overview
This document describes the unit testing approach for the ControlNet integration in the Stable Diffusion server. The tests are designed to use mocking extensively to avoid loading real models, making them fast and suitable for continuous integration.

## Test Structure

### 1. Image Processing Tests (`test_image_processing.py`)
- **Purpose**: Test utility functions for image processing
- **Coverage**: 
  - `get_stable_diffusion_size()` - Get appropriate dimensions for different aspect ratios
  - `get_closest_stable_diffusion_size()` - Find closest SD size for input dimensions
  - `aspect_ratio_upscale_and_crop()` - Resize and crop images while preserving aspect ratio
  - `process_image_for_stable_diffusion()` - Complete image preprocessing pipeline

### 2. Diffusion Utilities Tests (`test_diffusion_util.py`)
- **Purpose**: Test model loading and utility functions
- **Coverage**:
  - Model configuration validation
  - `load_flow_model()` with cache_dir support
  - `load_t5()` and `load_clip()` embedder loading
  - `load_ae()` autoencoder loading
  - `load_image()` image preprocessing
  - `embed_watermark()` watermark functionality

### 3. CustomPipeline Tests (`test_custom_pipeline.py`)
- **Purpose**: Test the main ControlNet pipeline
- **Coverage**:
  - Pipeline initialization with cache_dir
  - ControlNet loading from safetensors
  - Image generation without control image
  - Image generation with control image
  - Error handling for missing dependencies

### 4. ControlNet Integration Tests (`test_controlnet_integration.py`)
- **Purpose**: End-to-end testing of ControlNet functionality
- **Coverage**:
  - Complete pipeline workflow
  - Integration between components
  - Mock-based generation testing
  - Configuration validation

## Key Testing Patterns

### 1. Heavy Dependency Mocking
```python
@patch('stable_diffusion_server.custom_pipeline.load_t5')
@patch('stable_diffusion_server.custom_pipeline.load_clip')
@patch('stable_diffusion_server.custom_pipeline.load_flow_model')
@patch('stable_diffusion_server.custom_pipeline.load_ae')
def test_pipeline_init(self, mock_load_ae, mock_load_flow, mock_load_clip, mock_load_t5):
    # Test pipeline initialization without loading real models
```

### 2. Torch Operations Mocking
```python
with patch('torch.Generator') as mock_gen:
    with patch('torch.randn') as mock_randn:
        # Test tensor operations without GPU requirements
```

### 3. File I/O Mocking
```python
with patch('PIL.Image.open') as mock_open:
    with patch('io.BytesIO') as mock_bio:
        # Test file operations without real files
```

### 4. Configuration Testing
```python
def test_model_configs():
    from stable_diffusion_server.diffusion_util import configs
    assert "flux-schnell" in configs
    assert configs["flux-schnell"].repo_id == "black-forest-labs/FLUX.1-schnell"
```

## Running Tests

### Run All Unit Tests
```bash
python -m pytest tests/unit/ -v
```

### Run Specific Test Categories
```bash
# Image processing only
python -m pytest tests/unit/test_image_processing.py -v

# ControlNet integration only
python -m pytest tests/unit/test_controlnet_integration.py -v

# Diffusion utilities only
python -m pytest tests/unit/test_diffusion_util.py -v
```

### Run Individual Tests
```bash
# Test specific function
python -m pytest tests/unit/test_image_processing.py::test_get_stable_diffusion_size -v

# Test specific class
python -m pytest tests/unit/test_controlnet_integration.py::TestControlNetIntegration -v
```

## Test Environment Setup

### Prerequisites
- Python 3.12+
- Virtual environment activated
- Dependencies installed (pytest, unittest.mock, PIL, torch)

### Environment Variables
Tests use mocked environment variables:
- `BUCKET_NAME`: test-bucket
- `BUCKET_PATH`: test-path  
- `STORAGE_PROVIDER`: r2
- `HF_HOME`: /tmp/test_cache

## Benefits of This Approach

### 1. **Fast Execution**
- No model downloads or GPU operations
- Tests run in seconds instead of minutes
- Suitable for CI/CD pipelines

### 2. **Isolated Testing**
- Each component tested independently
- No external dependencies required
- Predictable test outcomes

### 3. **Comprehensive Coverage**
- Tests cover error conditions
- Edge cases are easily simulated
- Integration paths are validated

### 4. **Development Friendly**
- Tests can run without GPU
- No need for large model files
- Easy to add new test cases

## Integration with CI/CD

The unit tests are designed to run in any environment:
- No GPU required
- No model downloads
- Minimal memory usage
- Fast execution time

This makes them ideal for:
- Pre-commit hooks
- Pull request validation
- Continuous integration
- Development environment testing

## Future Enhancements

1. **Performance Testing**: Add benchmarks for key functions
2. **Memory Usage Testing**: Validate memory cleanup in pipelines
3. **Error Injection Testing**: Test resilience to various failure modes
4. **Configuration Validation**: Ensure all model configs are valid
5. **API Endpoint Testing**: Test FastAPI endpoints with mocked backends

## Conclusion

The unit testing framework provides comprehensive coverage of the ControlNet functionality while maintaining fast execution times through extensive mocking. This approach allows developers to validate their changes quickly without the overhead of loading actual AI models, making it practical for everyday development workflows.
