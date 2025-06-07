import os
import sys
import importlib
from unittest.mock import patch, MagicMock

with patch.dict(sys.modules, {
    'cv2': MagicMock(__spec__=MagicMock()),
    'google': MagicMock(),
    'google.cloud': MagicMock(),
}):
    with patch('diffusers.DiffusionPipeline.from_pretrained', return_value=MagicMock()) as _:
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
