import pytest
from PIL import Image
from stable_diffusion_server.image_processing import (
    aspect_ratio_upscale_and_crop,
    get_stable_diffusion_size,
    process_image_for_stable_diffusion,
    get_closest_stable_diffusion_size,
    process_image_for_stable_diffusion_ar
)

@pytest.fixture
def castle_image():
    return Image.open("tests/castlesketch.jpg")

def test_aspect_ratio_upscale_and_crop(castle_image):
    target_size = (1024, 1024)
    result = aspect_ratio_upscale_and_crop(castle_image, target_size)
    assert result.size == target_size

def test_get_stable_diffusion_size():
    assert get_stable_diffusion_size("1:1") == (1024, 1024)
    assert get_stable_diffusion_size("16:9") == (1360, 768)
    assert get_stable_diffusion_size("invalid") == (1024, 1024)  # Default case

def test_process_image_for_stable_diffusion_with_aspect_ratio(castle_image):
    result = process_image_for_stable_diffusion_ar(castle_image, "16:9")
    assert result.size == (1360, 768)

def test_get_closest_stable_diffusion_size():
    closest_size = get_closest_stable_diffusion_size(1000, 750)
    assert closest_size in [(1024, 1024), (1152, 768), (1152, 864)]

def test_get_closest_stable_diffusion_size_2():
    closest_size = get_closest_stable_diffusion_size(2048, 2048)
    assert closest_size in [(1024, 1024), (1152, 768), (1152, 864)]

def test_process_image_for_stable_diffusion_without_aspect_ratio(castle_image):
    result = process_image_for_stable_diffusion(castle_image)
    assert result.size in [(1024, 1024), (1152, 768), (768, 1152), (1152, 864), (864, 1152), (1360, 768), (768, 1360)]
