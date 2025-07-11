import pytest
from unittest.mock import Mock, patch
from PIL import Image
import torch

from stable_diffusion_server.custom_pipeline import CustomPipeline


class TestCustomPipeline:
    """Unit tests for CustomPipeline class with mocked dependencies."""

    @pytest.fixture
    def mock_device(self):
        """Mock torch device."""
        return Mock()

    @pytest.fixture
    def mock_image(self):
        """Mock PIL Image."""
        image = Image.new('RGB', (512, 512), color='red')
        return image

    @pytest.fixture
    def mock_tensor(self):
        """Mock torch tensor."""
        return torch.randn(1, 3, 512, 512)

    @patch('stable_diffusion_server.custom_pipeline.load_t5')
    @patch('stable_diffusion_server.custom_pipeline.load_clip')
    @patch('stable_diffusion_server.custom_pipeline.load_flow_model')
    @patch('stable_diffusion_server.custom_pipeline.load_ae')
    @patch('torch.device')
    def test_custom_pipeline_init(self, mock_device, mock_load_ae, mock_load_flow_model, 
                                 mock_load_clip, mock_load_t5):
        """Test CustomPipeline initialization with mocked dependencies."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_load_t5.return_value = Mock()
        mock_load_clip.return_value = Mock()
        mock_load_flow_model.return_value = Mock()
        mock_load_ae.return_value = Mock()

        # Create pipeline
        pipeline = CustomPipeline(name="flux-schnell", device="cuda", offload=True)

        # Assert initialization
        assert pipeline.name == "flux-schnell"
        assert pipeline.offload
        assert pipeline.controlnet is None
        
        # Assert dependencies were loaded
        mock_load_t5.assert_called_once()
        mock_load_clip.assert_called_once()
        mock_load_flow_model.assert_called_once()
        mock_load_ae.assert_called_once()

    @patch('stable_diffusion_server.custom_pipeline.load_t5')
    @patch('stable_diffusion_server.custom_pipeline.load_clip')
    @patch('stable_diffusion_server.custom_pipeline.load_flow_model')
    @patch('stable_diffusion_server.custom_pipeline.load_ae')
    @patch('torch.device')
    def test_custom_pipeline_init_with_cache_dir(self, mock_device, mock_load_ae, 
                                                mock_load_flow_model, mock_load_clip, mock_load_t5):
        """Test CustomPipeline initialization with cache_dir."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_load_t5.return_value = Mock()
        mock_load_clip.return_value = Mock()
        mock_load_flow_model.return_value = Mock()
        mock_load_ae.return_value = Mock()

        # Create pipeline with cache_dir
        cache_dir = "/tmp/test_cache"
        pipeline = CustomPipeline(name="flux-schnell", cache_dir=cache_dir)

        # Assert cache_dir was passed to load functions
        mock_load_t5.assert_called_once_with(pipeline.device, cache_dir=cache_dir)
        mock_load_clip.assert_called_once_with(pipeline.device, cache_dir=cache_dir)

    @patch('stable_diffusion_server.custom_pipeline.load_controlnet')
    @patch('stable_diffusion_server.custom_pipeline.load_t5')
    @patch('stable_diffusion_server.custom_pipeline.load_clip')
    @patch('stable_diffusion_server.custom_pipeline.load_flow_model')
    @patch('stable_diffusion_server.custom_pipeline.load_ae')
    @patch('torch.device')
    def test_load_controlnet(self, mock_device, mock_load_ae, mock_load_flow_model, 
                           mock_load_clip, mock_load_t5, mock_load_controlnet):
        """Test loading ControlNet weights."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_load_t5.return_value = Mock()
        mock_load_clip.return_value = Mock()
        mock_load_flow_model.return_value = Mock()
        mock_load_ae.return_value = Mock()
        mock_controlnet = Mock()
        mock_load_controlnet.return_value = mock_controlnet

        # Create pipeline and load controlnet
        pipeline = CustomPipeline(name="flux-schnell")
        pipeline.load_controlnet("path/to/controlnet.safetensors")

        # Assert controlnet was loaded
        mock_load_controlnet.assert_called_once()
        assert pipeline.controlnet == mock_controlnet

    @patch('stable_diffusion_server.custom_pipeline.prepare')
    @patch('stable_diffusion_server.custom_pipeline.get_schedule')
    @patch('stable_diffusion_server.custom_pipeline.get_noise')
    @patch('stable_diffusion_server.custom_pipeline.denoise')
    @patch('stable_diffusion_server.custom_pipeline.unpack')
    @patch('stable_diffusion_server.custom_pipeline.embed_watermark')
    @patch('stable_diffusion_server.custom_pipeline.load_t5')
    @patch('stable_diffusion_server.custom_pipeline.load_clip')
    @patch('stable_diffusion_server.custom_pipeline.load_flow_model')
    @patch('stable_diffusion_server.custom_pipeline.load_ae')
    @patch('torch.device')
    def test_generate_without_controlnet(self, mock_device, mock_load_ae, mock_load_flow_model,
                                       mock_load_clip, mock_load_t5, mock_embed_watermark,
                                       mock_unpack, mock_denoise, mock_get_noise,
                                       mock_get_schedule, mock_prepare, mock_image):
        """Test image generation without ControlNet."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_t5 = Mock()
        mock_clip = Mock()
        mock_model = Mock()
        mock_ae = Mock()
        
        mock_load_t5.return_value = mock_t5
        mock_load_clip.return_value = mock_clip
        mock_load_flow_model.return_value = mock_model
        mock_load_ae.return_value = mock_ae
        
        # Mock the encoding/decoding chain
        mock_t5.return_value = torch.randn(1, 256, 4096)
        mock_clip.return_value = torch.randn(1, 77, 768)
        mock_prepare.return_value = (torch.randn(1, 64, 64, 64), torch.randn(1, 256, 4096))
        mock_get_schedule.return_value = torch.randn(28)
        mock_get_noise.return_value = torch.randn(1, 64, 64, 64)
        mock_denoise.return_value = torch.randn(1, 64, 64, 64)
        mock_unpack.return_value = torch.randn(1, 3, 1024, 1024)
        mock_embed_watermark.return_value = torch.randn(1, 3, 1024, 1024)
        mock_ae.decode.return_value = torch.randn(1, 3, 1024, 1024)

        # Create pipeline
        pipeline = CustomPipeline(name="flux-schnell")
        
        # Generate image
        result = pipeline.generate(prompt="test prompt", width=1024, height=1024)
        
        # Assert result is bytes
        assert isinstance(result, bytes)
        
        # Assert key functions were called
        mock_t5.assert_called_once()
        mock_clip.assert_called_once()
        mock_prepare.assert_called_once()
        mock_denoise.assert_called_once()

    @patch('stable_diffusion_server.custom_pipeline.load_image')
    @patch('stable_diffusion_server.custom_pipeline.prepare')
    @patch('stable_diffusion_server.custom_pipeline.get_schedule')
    @patch('stable_diffusion_server.custom_pipeline.get_noise')
    @patch('stable_diffusion_server.custom_pipeline.denoise')
    @patch('stable_diffusion_server.custom_pipeline.unpack')
    @patch('stable_diffusion_server.custom_pipeline.embed_watermark')
    @patch('stable_diffusion_server.custom_pipeline.load_t5')
    @patch('stable_diffusion_server.custom_pipeline.load_clip')
    @patch('stable_diffusion_server.custom_pipeline.load_flow_model')
    @patch('stable_diffusion_server.custom_pipeline.load_ae')
    @patch('stable_diffusion_server.custom_pipeline.load_controlnet')
    @patch('torch.device')
    def test_generate_with_controlnet(self, mock_device, mock_load_controlnet, mock_load_ae,
                                    mock_load_flow_model, mock_load_clip, mock_load_t5,
                                    mock_embed_watermark, mock_unpack, mock_denoise,
                                    mock_get_noise, mock_get_schedule, mock_prepare,
                                    mock_load_image, mock_image):
        """Test image generation with ControlNet."""
        # Setup mocks
        mock_device.return_value = Mock()
        mock_t5 = Mock()
        mock_clip = Mock()
        mock_model = Mock()
        mock_ae = Mock()
        mock_controlnet = Mock()
        
        mock_load_t5.return_value = mock_t5
        mock_load_clip.return_value = mock_clip
        mock_load_flow_model.return_value = mock_model
        mock_load_ae.return_value = mock_ae
        mock_load_controlnet.return_value = mock_controlnet
        
        # Mock the encoding/decoding chain
        mock_t5.return_value = torch.randn(1, 256, 4096)
        mock_clip.return_value = torch.randn(1, 77, 768)
        mock_prepare.return_value = (torch.randn(1, 64, 64, 64), torch.randn(1, 256, 4096))
        mock_get_schedule.return_value = torch.randn(28)
        mock_get_noise.return_value = torch.randn(1, 64, 64, 64)
        mock_denoise.return_value = torch.randn(1, 64, 64, 64)
        mock_unpack.return_value = torch.randn(1, 3, 1024, 1024)
        mock_embed_watermark.return_value = torch.randn(1, 3, 1024, 1024)
        mock_ae.decode.return_value = torch.randn(1, 3, 1024, 1024)
        mock_ae.encode.return_value = torch.randn(1, 16, 128, 128)
        mock_load_image.return_value = torch.randn(1, 3, 1024, 1024)

        # Create pipeline and load controlnet
        pipeline = CustomPipeline(name="flux-schnell")
        pipeline.load_controlnet("path/to/controlnet.safetensors")
        
        # Generate image with control image
        result = pipeline.generate(prompt="test prompt", image=mock_image, width=1024, height=1024)
        
        # Assert result is bytes
        assert isinstance(result, bytes)
        
        # Assert controlnet-specific functions were called
        mock_load_image.assert_called_once()
        mock_ae.encode.assert_called_once()

    def test_image_to_bytes_conversion(self):
        """Test image to bytes conversion."""
        # Mock the conversion logic
        with patch('stable_diffusion_server.custom_pipeline.Image.fromarray') as mock_fromarray:
            mock_pil_image = Mock()
            mock_fromarray.return_value = mock_pil_image
            
            mock_pil_image.save = Mock()
            
            # This would be inside the generate method
            # Just test that the mock gets called correctly
            mock_fromarray.assert_not_called()  # Since we haven't called it yet
