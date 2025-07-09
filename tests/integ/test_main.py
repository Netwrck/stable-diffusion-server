# import pytest
# pytest.skip(reason="requires heavy stable diffusion models", allow_module_level=True)

from PIL import Image

import pillow_avif
assert pillow_avif

from main import (
    create_image_from_prompt,
    inpaint_image_from_prompt,
    style_transfer_image_from_prompt,
)


def test_create_image_from_prompt_sync():
    imagebytesresult = create_image_from_prompt("a test prompt", 512, 512)
    assert imagebytesresult is not None


def test_create_image_from_prompt_sync_bug():
    imagebytesresult = create_image_from_prompt(
        "artstation art art paint colorful swirl letter a a paint colorful swirl confident engaging wow",
        1024,
        1024,
        n_steps=20)
    assert imagebytesresult is not None
    # save to disk
    with open("results/testcreateimage.webp", "wb") as f:
        f.write(imagebytesresult)


def test_inpaint_from_prompt_sync():
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    imagebytesresult = inpaint_image_from_prompt("a test prompt", img_url,
                                                 mask_url)

    assert imagebytesresult is not None


def test_style_transfer_from_prompt_sync():
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    style_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/style_transfer_examples/2.jpg"

    image_bytes = style_transfer_image_from_prompt("a lion", img_url, 0.6)
    # save to disk
    
    with open("results/teststyletransfer.webp", "wb") as f:
        f.write(image_bytes)

    assert image_bytes is not None


def test_style_transfer_from_prompt_sync_controlnet():
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"

    image_bytes = style_transfer_image_from_prompt("a lion", img_url, 0.6,
                                                      True)
    # save to disk
    with open("results/teststyletransfer-cnet.webp", "wb") as f:
        f.write(image_bytes)

    assert image_bytes is not None


def test_style_transfer_from_prompt_sync_control_pil():
    img = Image.open("tests/castlesketch-big.jpg").convert("RGB")
    # img = Image.open("tests/owl.png").convert("RGB")

    image_bytes = style_transfer_image_from_prompt(
        "a owl cinematic wonderful realistic owl perching on log", None, 0.6,
        True, img)
    # save to disk

    with open("results/teststyletransfer-cnet-owl.webp", "wb") as f:
        f.write(image_bytes)

    assert image_bytes is not None



def test_style_transfer_from_prompt_sync_control_pil_dude():
    img = Image.open("tests/dudehappy.jpeg").convert("RGB")
    # img = Image.open("tests/owl.png").convert("RGB")

    image_bytes = style_transfer_image_from_prompt(
        "starry night van gogh", None, 0.6,
        True, img)
    # save to disk

    with open("results/teststyletransfer-cnet-starry.webp", "wb") as f:
        f.write(image_bytes)

    assert image_bytes is not None


def test_style_transfer_from_prompt_sync_control_avif():
    img = Image.open("tests/andrew-ng.avif").convert("RGB")
    # img = Image.open("tests/owl.png").convert("RGB")

    image_bytes = style_transfer_image_from_prompt(
        "a owl cinematic wonderful realistic owl perching on log", None, 0.6,
        True, img)
    # save to disk

    with open("results/teststyletransfer-cnet-owl-again.webp", "wb") as f:
        f.write(image_bytes)

    assert image_bytes is not None
