from PIL import Image

from main import (
    create_image_from_prompt,
    inpaint_image_from_prompt,
    style_transfer_image_from_prompt,
)


def test_create_image_from_prompt_sync():
    imagebytesresult = create_image_from_prompt("a test prompt", 512, 512)
    assert imagebytesresult is not None

def test_create_image_from_prompt_sync_bug():
    imagebytesresult = create_image_from_prompt("artstation art art paint colorful swirl letter a a paint colorful swirl confident engaging wow", 1024, 1024, n_steps=20)
    assert imagebytesresult is not None
    # save to disk
    with open("testcreateimage.webp", "wb") as f:
        f.write(imagebytesresult)


def test_inpaint_from_prompt_sync():
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    imagebytesresult = inpaint_image_from_prompt("a test prompt", img_url, mask_url)

    assert imagebytesresult is not None


def test_style_transfer_from_prompt_sync():
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    style_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/style_transfer_examples/2.jpg"

    imagepilresult = style_transfer_image_from_prompt("a lion", img_url, 0.6)
    # save to disk
    imagepilresult.save("teststyletransfer.png")

    assert imagepilresult is not None


def test_style_transfer_from_prompt_sync_controlnet():
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"

    imagepilresult = style_transfer_image_from_prompt("a lion", img_url, 0.6, True)
    # save to disk
    imagepilresult.save("teststyletransfer-cnet.png")

    assert imagepilresult is not None


def test_style_transfer_from_prompt_sync_control_pil():
    img = Image.open("tests/castlesketch-big.jpg").convert("RGB")
    img = Image.open("tests/owl.png").convert("RGB")

    imagepilresult = style_transfer_image_from_prompt(
        "a owl cinematic wonderful realistic owl perching on log", None, 0.6, True, img
    )
    # save to disk
    imagepilresult.save("teststyletransfer-cnet-tmp-owl.png")

    assert imagepilresult is not None

def test_style_transfer_from_prompt_sync_control_avif():
    img = Image.open("tests/andrew-ng.avif").convert("RGB")
    img = Image.open("tests/owl.png").convert("RGB")

    imagepilresult = style_transfer_image_from_prompt(
        "a owl cinematic wonderful realistic owl perching on log", None, 0.6, True, img
    )
    # save to disk
    imagepilresult.save("teststyletransfer-cnet-tmp-owl.png")

    assert imagepilresult is not None
