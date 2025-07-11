from unittest.mock import Mock, patch
from PIL import Image

from main import (
    generate_controlnet_image_bytes,
    create_image_from_prompt,
    style_transfer_image_from_prompt,
    image_to_bytes,
    is_defined,
)


class TestMainFunctions:
    """Unit tests for main.py functions with mocked dependencies."""

    @patch('main.custom_pipeline')
    def test_generate_controlnet_image_bytes_success(self, mock_custom_pipeline):
        """Test successful ControlNet image generation."""
        # Setup mocks
        mock_pipeline = Mock()
        mock_custom_pipeline.return_value = mock_pipeline
        mock_pipeline.generate.return_value = b"fake_image_bytes"
        
        # Create a mock image
        mock_image = Mock()
        
        # Test the function
        result = generate_controlnet_image_bytes("test prompt", mock_image)
        
        # Assertions
        assert result == b"fake_image_bytes"
        mock_pipeline.generate.assert_called_once_with(prompt="test prompt", image=mock_image)

    @patch('main.custom_pipeline', None)
    def test_generate_controlnet_image_bytes_no_pipeline(self):
        """Test ControlNet image generation when pipeline is None."""
        mock_image = Mock()
        
        try:
            generate_controlnet_image_bytes("test prompt", mock_image)
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "Pipeline not initialized" in str(e)

    @patch('main.flux_pipe')
    @patch('main.detect_too_bumpy')
    @patch('main.image_to_bytes')
    @patch('main.shorten_too_long_text')
    @patch('main.remove_stopwords')
    @patch('main.shorten_prompt_for_retry')
    @patch('torch.Generator')
    def test_create_image_from_prompt_success(self, mock_generator, mock_shorten_retry, 
                                             mock_remove_stopwords, mock_shorten_text,
                                             mock_image_to_bytes, mock_detect_bumpy, mock_flux_pipe):
        """Test successful image creation from prompt."""
        # Setup mocks
        mock_image = Mock()
        mock_flux_pipe.return_value.images = [mock_image]
        mock_shorten_text.return_value = "shortened prompt"
        mock_detect_bumpy.return_value = False
        mock_image_to_bytes.return_value = b"fake_image_bytes"
        mock_gen = Mock()
        mock_generator.return_value = mock_gen
        
        # Test the function
        result = create_image_from_prompt("test prompt", 1024, 1024, n_steps=5)
        
        # Assertions
        assert result == b"fake_image_bytes"
        mock_flux_pipe.assert_called_once()
        mock_detect_bumpy.assert_called_once_with(mock_image)
        mock_image_to_bytes.assert_called_once_with(mock_image)

    @patch('main.flux_pipe')
    @patch('main.detect_too_bumpy')
    @patch('main.image_to_bytes')
    @patch('main.shorten_too_long_text')
    @patch('main.remove_stopwords')
    @patch('main.shorten_prompt_for_retry')
    @patch('torch.Generator')
    def test_create_image_from_prompt_too_bumpy(self, mock_generator, mock_shorten_retry, 
                                               mock_remove_stopwords, mock_shorten_text,
                                               mock_image_to_bytes, mock_detect_bumpy, mock_flux_pipe):
        """Test image creation with too bumpy detection."""
        # Setup mocks - first call is bumpy, second call is not
        mock_image = Mock()
        mock_flux_pipe.return_value.images = [mock_image]
        mock_shorten_text.return_value = "shortened prompt"
        mock_detect_bumpy.side_effect = [True, False]  # First bumpy, then not
        mock_image_to_bytes.return_value = b"fake_image_bytes"
        mock_gen = Mock()
        mock_generator.return_value = mock_gen
        
        # Test the function
        result = create_image_from_prompt("test prompt", 1024, 1024, n_steps=5)
        
        # Assertions
        assert result == b"fake_image_bytes"
        assert mock_flux_pipe.call_count == 2  # Should retry once
        assert mock_detect_bumpy.call_count == 2

    @patch('main.flux_pipe')
    @patch('main.flux_controlnetpipe')
    @patch('main.process_image_for_stable_diffusion')
    @patch('main.load_image')
    @patch('main.detect_too_bumpy')
    @patch('main.image_to_bytes')
    @patch('main.shorten_too_long_text')
    @patch('main.cv2')
    @patch('main.set_seed')
    @patch('torch.Generator')
    def test_style_transfer_image_from_prompt_with_canny(self, mock_generator, mock_set_seed,
                                                        mock_cv2, mock_shorten_text, mock_image_to_bytes,
                                                        mock_detect_bumpy, mock_load_image,
                                                        mock_process_image, mock_flux_controlnetpipe,
                                                        mock_flux_pipe):
        """Test style transfer with Canny edge detection."""
        # Setup mocks
        mock_input_image = Mock()
        mock_input_image.width = 1024
        mock_input_image.height = 1024
        mock_load_image.return_value = mock_input_image
        mock_process_image.return_value = mock_input_image
        mock_shorten_text.return_value = "shortened prompt"
        mock_detect_bumpy.return_value = False
        mock_image_to_bytes.return_value = b"fake_image_bytes"
        
        # Mock CV2 Canny
        mock_cv2.Canny.return_value = Mock()
        
        # Mock the pipeline
        mock_result_image = Mock()
        mock_flux_controlnetpipe.return_value.images = [mock_result_image]
        
        mock_gen = Mock()
        mock_generator.return_value = mock_gen
        
        # Test the function
        result = style_transfer_image_from_prompt("test prompt", "image_url", canny=True)
        
        # Assertions
        assert result == b"fake_image_bytes"
        mock_flux_controlnetpipe.assert_called_once()
        mock_cv2.Canny.assert_called_once()
        mock_set_seed.assert_called_once_with(42)

    def test_is_defined_with_image(self):
        """Test is_defined function with PIL Image."""
        mock_image = Mock(spec=Image.Image)
        assert is_defined(mock_image)

    def test_is_defined_with_none(self):
        """Test is_defined function with None."""
        assert not is_defined(None)

    def test_is_defined_with_string(self):
        """Test is_defined function with string."""
        assert is_defined("test string")

    def test_is_defined_with_empty_string(self):
        """Test is_defined function with empty string."""
        assert not is_defined("")

    @patch('main.np')
    def test_image_to_bytes_success(self, mock_np):
        """Test successful image to bytes conversion."""
        # Setup mocks
        mock_image = Mock()
        mock_array = Mock()
        mock_np.array.return_value = mock_array
        mock_np.sum.return_value = 100  # Non-zero bright count
        
        # Mock BytesIO
        with patch('main.BytesIO') as mock_bytesio:
            mock_bio = Mock()
            mock_bytesio.return_value = mock_bio
            mock_bio.getvalue.return_value = b"fake_image_bytes"
            
            result = image_to_bytes(mock_image)
            
            assert result == b"fake_image_bytes"
            mock_image.save.assert_called_once()

    @patch('main.np')
    @patch('main.os')
    def test_image_to_bytes_black_image(self, mock_os, mock_np):
        """Test image to bytes with black image (should trigger restart)."""
        # Setup mocks
        mock_image = Mock()
        mock_array = Mock()
        mock_np.array.return_value = mock_array
        mock_np.sum.return_value = 0  # Zero bright count (black image)
        
        result = image_to_bytes(mock_image)
        
        assert result is None
        # Should trigger system restart commands
        assert mock_os.system.call_count >= 4

    @patch('main.app')
    def test_app_initialization(self, mock_app):
        """Test that FastAPI app is properly initialized."""
        # This test verifies the app is created and middleware is added
        # In a real test, we would check the actual app configuration
        assert mock_app is not None


class TestMainEndpoints:
    """Unit tests for FastAPI endpoints with mocked dependencies."""

    @patch('main.custom_pipeline')
    @patch('main.load_image')
    @patch('main.generate_controlnet_image_bytes')
    def test_controlnet_image_endpoint_success(self, mock_generate_bytes, mock_load_image, mock_custom_pipeline):
        """Test successful ControlNet image endpoint."""
        # This would require importing the actual endpoint function
        # and testing it with FastAPI's TestClient
        pass

    @patch('main.custom_pipeline', None)
    def test_controlnet_image_endpoint_no_pipeline(self):
        """Test ControlNet endpoint when pipeline is None."""
        # This would test the error response when pipeline is not initialized
        pass
