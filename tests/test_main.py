from main import create_image_from_prompt, inpaint_image_from_prompt, style_transfer_image_from_prompt


def test_create_image_from_prompt_sync():
    imagebytesresult = create_image_from_prompt("a test prompt", 512, 512)
    assert imagebytesresult is not None

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
