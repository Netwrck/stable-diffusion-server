import os
import pytest
from PIL import Image

try:
    from gradio_inpaint_ui import _save_temp_image
except Exception:
    pytest.skip("gradio not available", allow_module_level=True)


def test_save_temp_image(tmp_path):
    img = Image.new("RGB", (10, 10), color="red")
    path = _save_temp_image(img, "test")
    assert os.path.exists(path)
    assert path.endswith(".png")
    os.remove(path)
