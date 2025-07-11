from unittest.mock import Mock, patch
import torch
import os
from PIL import Image

from stable_diffusion_server.diffusion_util import (
    load_flow_model,
    load_t5,
    load_clip,
    load_ae,
    load_image,
    embed_watermark,
    configs,
)


class TestDiffusionUtil:
    """Unit tests for diffusion utility functions with mocked dependencies."""

    def test_configs_exist(self):
        """Test that model configurations exist."""
        assert "flux-schnell" in configs
        assert "flux-dev" in configs
        assert "flux-dev-fp8" in configs
        
        # Test flux-schnell config
        flux_schnell_config = configs["flux-schnell"]
        assert flux_schnell_config.repo_id == "black-forest-labs/FLUX.1-schnell"
        assert flux_schnell_config.params.guidance_embed is False

    @patch('stable_diffusion_server.diffusion_util.hf_hub_download')
    @patch('stable_diffusion_server.diffusion_util.load_sft')
    @patch('stable_diffusion_server.diffusion_util.Flux')
    @patch('torch.device')
    def test_load_flow_model_with_download(self, mock_device, mock_flux, mock_load_sft, mock_hf_download):
        """Test loading flow model with Hub download."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_model = Mock()
        mock_flux.return_value = mock_model
        mock_hf_download.return_value = "/tmp/cached_model.safetensors"
        mock_load_sft.return_value = {"key": "value"}
        mock_model.load_state_dict.return_value = ([], [])

        # Test loading
        result = load_flow_model("flux-schnell", device="cuda", cache_dir="/tmp/cache")
        
        # Assertions
        assert result == mock_model
        mock_hf_download.assert_called_once()
        mock_load_sft.assert_called_once()
        mock_model.load_state_dict.assert_called_once()

    @patch('stable_diffusion_server.diffusion_util.hf_hub_download')
    @patch('stable_diffusion_server.diffusion_util.load_sft')
    @patch('stable_diffusion_server.diffusion_util.Flux')
    @patch('torch.device')
    def test_load_flow_model_with_local_path(self, mock_device, mock_flux, mock_load_sft, mock_hf_download):
        """Test loading flow model with local path."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_model = Mock()
        mock_flux.return_value = mock_model
        mock_load_sft.return_value = {"key": "value"}
        mock_model.load_state_dict.return_value = ([], [])

        # Set environment variable to simulate local path
        with patch.dict(os.environ, {'FLUX_SCHNELL': '/local/path/model.safetensors'}):
            result = load_flow_model("flux-schnell", device="cuda")
        
        # Assertions
        assert result == mock_model
        mock_hf_download.assert_not_called()  # Should not download if local path exists
        mock_load_sft.assert_called_once_with('/local/path/model.safetensors', device="cuda")

    @patch('stable_diffusion_server.diffusion_util.HFEmbedder')
    def test_load_t5(self, mock_embedder):
        """Test loading T5 embedder."""
        mock_t5 = Mock()
        mock_embedder.return_value = mock_t5
        mock_t5.to.return_value = mock_t5

        result = load_t5(device="cuda", cache_dir="/tmp/cache")
        
        assert result == mock_t5
        mock_embedder.assert_called_once_with(
            "google/t5-v1_1-xxl", 
            max_length=512, 
            torch_dtype=torch.bfloat16, 
            cache_dir="/tmp/cache"
        )
        mock_t5.to.assert_called_once_with("cuda")

    @patch('stable_diffusion_server.diffusion_util.HFEmbedder')
    def test_load_clip(self, mock_embedder):
        """Test loading CLIP embedder."""
        mock_clip = Mock()
        mock_embedder.return_value = mock_clip
        mock_clip.to.return_value = mock_clip

        result = load_clip(device="cuda", cache_dir="/tmp/cache")
        
        assert result == mock_clip
        mock_embedder.assert_called_once_with(
            "openai/clip-vit-large-patch14",
            max_length=77,
            torch_dtype=torch.bfloat16,
            cache_dir="/tmp/cache"
        )
        mock_clip.to.assert_called_once_with("cuda")

    @patch('stable_diffusion_server.diffusion_util.hf_hub_download')
    @patch('stable_diffusion_server.diffusion_util.load_sft')
    @patch('stable_diffusion_server.diffusion_util.AutoEncoder')
    @patch('torch.device')
    def test_load_ae(self, mock_device, mock_autoencoder, mock_load_sft, mock_hf_download):
        """Test loading AutoEncoder."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_ae = Mock()
        mock_autoencoder.return_value = mock_ae
        mock_hf_download.return_value = "/tmp/cached_ae.safetensors"
        mock_load_sft.return_value = {"key": "value"}
        mock_ae.load_state_dict.return_value = ([], [])

        result = load_ae("flux-schnell", device="cuda", cache_dir="/tmp/cache")
        
        assert result == mock_ae
        mock_hf_download.assert_called_once()
        mock_load_sft.assert_called_once()
        mock_ae.load_state_dict.assert_called_once()

    def test_load_image_from_path(self):
        """Test loading image from file path."""
        # Mock PIL Image
        with patch('stable_diffusion_server.diffusion_util.PILImage') as mock_pil:
            mock_image = Mock()
            mock_pil.open.return_value = mock_image
            mock_image.mode = 'RGB'
            mock_image.resize.return_value = mock_image
            
            # Mock numpy and torch operations
            with patch('stable_diffusion_server.diffusion_util.np') as mock_np:
                with patch('stable_diffusion_server.diffusion_util.torch') as mock_torch:
                    mock_array = Mock()
                    mock_np.array.return_value = mock_array
                    mock_array.astype.return_value = mock_array
                    
                    mock_tensor = Mock()
                    mock_torch.from_numpy.return_value = mock_tensor
                    mock_tensor.permute.return_value = mock_tensor
                    mock_tensor.unsqueeze.return_value = mock_tensor
                    
                    result = load_image("/path/to/image.jpg", height=512, width=512)
                    
                    assert result == mock_tensor
                    mock_pil.open.assert_called_once_with("/path/to/image.jpg")
                    mock_image.resize.assert_called_once_with((512, 512), mock_pil.LANCZOS)

    def test_load_image_from_pil(self):
        """Test loading image from PIL Image."""
        # Create a mock PIL Image
        mock_image = Mock()
        mock_image.mode = 'RGB'
        mock_image.resize.return_value = mock_image
        
        # Mock numpy and torch operations
        with patch('stable_diffusion_server.diffusion_util.np') as mock_np:
            with patch('stable_diffusion_server.diffusion_util.torch') as mock_torch:
                mock_array = Mock()
                mock_np.array.return_value = mock_array
                mock_array.astype.return_value = mock_array
                
                mock_tensor = Mock()
                mock_torch.from_numpy.return_value = mock_tensor
                mock_tensor.permute.return_value = mock_tensor
                mock_tensor.unsqueeze.return_value = mock_tensor
                
                result = load_image(mock_image, height=512, width=512)
                
                assert result == mock_tensor
                mock_image.resize.assert_called_once_with((512, 512), Image.LANCZOS)

    def test_load_image_convert_mode(self):
        """Test loading image with mode conversion."""
        # Mock PIL Image with RGBA mode
        mock_image = Mock()
        mock_image.mode = 'RGBA'
        mock_converted_image = Mock()
        mock_converted_image.mode = 'RGB'
        mock_image.convert.return_value = mock_converted_image
        mock_converted_image.resize.return_value = mock_converted_image
        
        # Mock numpy and torch operations
        with patch('stable_diffusion_server.diffusion_util.np') as mock_np:
            with patch('stable_diffusion_server.diffusion_util.torch') as mock_torch:
                mock_array = Mock()
                mock_np.array.return_value = mock_array
                mock_array.astype.return_value = mock_array
                
                mock_tensor = Mock()
                mock_torch.from_numpy.return_value = mock_tensor
                mock_tensor.permute.return_value = mock_tensor
                mock_tensor.unsqueeze.return_value = mock_tensor
                
                result = load_image(mock_image, height=512, width=512)
                
                assert result == mock_tensor
                mock_image.convert.assert_called_once_with('RGB')

    def test_embed_watermark(self):
        """Test watermark embedding (currently a pass-through)."""
        mock_image = Mock()
        result = embed_watermark(mock_image)
        assert result == mock_image  # Should return unchanged for now

    @patch('stable_diffusion_server.diffusion_util.print')
    def test_print_load_warning_with_missing_and_unexpected(self, mock_print):
        """Test print_load_warning with both missing and unexpected keys."""
        from stable_diffusion_server.diffusion_util import print_load_warning
        
        missing = ["key1", "key2"]
        unexpected = ["key3", "key4"]
        
        print_load_warning(missing, unexpected)
        
        # Should print both missing and unexpected keys
        assert mock_print.call_count >= 2

    @patch('stable_diffusion_server.diffusion_util.print')
    def test_print_load_warning_with_only_missing(self, mock_print):
        """Test print_load_warning with only missing keys."""
        from stable_diffusion_server.diffusion_util import print_load_warning
        
        missing = ["key1", "key2"]
        unexpected = []
        
        print_load_warning(missing, unexpected)
        
        # Should print only missing keys
        assert mock_print.call_count >= 1

    @patch('stable_diffusion_server.diffusion_util.print')
    def test_print_load_warning_with_only_unexpected(self, mock_print):
        """Test print_load_warning with only unexpected keys."""
        from stable_diffusion_server.diffusion_util import print_load_warning
        
        missing = []
        unexpected = ["key3", "key4"]
        
        print_load_warning(missing, unexpected)
        
        # Should print only unexpected keys
        assert mock_print.call_count >= 1

    @patch('stable_diffusion_server.diffusion_util.print')
    def test_print_load_warning_with_no_keys(self, mock_print):
        """Test print_load_warning with no keys."""
        from stable_diffusion_server.diffusion_util import print_load_warning
        
        missing = []
        unexpected = []
        
        print_load_warning(missing, unexpected)
        
        # Should not print anything
        mock_print.assert_not_called()
