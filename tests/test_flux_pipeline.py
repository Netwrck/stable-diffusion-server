import os
import sys
import importlib
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with patch.dict(sys.modules, {
    'cv2': MagicMock(__spec__=MagicMock()),
    'google': MagicMock(),
    'google.cloud': MagicMock(),
    'optimum': MagicMock(__spec__=MagicMock()),
    'optimum.quanto': MagicMock(),
    'nltk': MagicMock(
        __spec__=MagicMock(),
        corpus=MagicMock(stopwords=MagicMock(words=lambda lang: [])),
    ),
}):
    with patch('diffusers.DiffusionPipeline.from_pretrained', return_value=MagicMock()) as _:
        with patch('diffusers.FluxPipeline.from_pretrained', return_value=MagicMock()) as _:
            with patch('diffusers.schedulers.scheduling_lcm.LCMScheduler.from_config', return_value=MagicMock()) as _:
                with patch('diffusers.AutoPipelineForImage2Image.from_pipe', return_value=MagicMock()) as _:
                    main = importlib.import_module('main')

@patch('main.upload_to_bucket')
@patch('main.check_if_blob_exists')
def test_get_flux_image_or_upload_existing(mock_exists, mock_upload):
    mock_exists.return_value = True
    result = main.get_flux_image_or_upload('test prompt', 'img.png')
    assert result == f"https://{main.BUCKET_NAME}/{main.BUCKET_PATH}/img.png"
    mock_upload.assert_not_called()

@patch('main.upload_to_bucket')
@patch('main.check_if_blob_exists')
def test_get_flux_image_or_upload_new(mock_exists, mock_upload):
    mock_exists.return_value = False
    mock_upload.return_value = 'url'
    fake_image = MagicMock()
    fake_image.images = [MagicMock()]
    with patch.object(main, 'flux_pipe') as mock_pipe:
        mock_pipe.return_value = fake_image
        with patch('main.image_to_bytes', return_value=b'x'):
            result = main.get_flux_image_or_upload('test', 'new.png')
    assert result == 'url'
    mock_upload.assert_called_once()


def test_flux_pipeline_disabled():
    with patch.dict(os.environ, {"ENABLE_FLUX_PIPELINE": "0"}):
        import importlib
        with patch.dict(sys.modules, {
            'cv2': MagicMock(__spec__=MagicMock()),
            'google': MagicMock(),
            'google.cloud': MagicMock(),
        }):
            with patch('diffusers.DiffusionPipeline.from_pretrained', return_value=MagicMock()):
                with patch('diffusers.FluxPipeline.from_pretrained', return_value=MagicMock()):
                    reloaded = importlib.reload(main)
        assert reloaded.flux_pipe is None


def test_create_flux_image_cache():
    fake_img = MagicMock()
    fake_img.images = [MagicMock()]
    with patch.object(main, 'flux_pipe', return_value=fake_img) as mock_pipe:
        with patch('main.image_to_bytes', return_value=b'x'):
            main._flux_cache.clear()
            res1 = main.create_flux_image_from_prompt('abc')
            res2 = main.create_flux_image_from_prompt('abc')
    assert res1 == res2
    assert mock_pipe.call_count == 1


def test_flux_health_endpoint():
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    with patch.object(main, 'flux_pipe', MagicMock()):
        resp = client.get('/flux_health')
    assert resp.status_code == 200
