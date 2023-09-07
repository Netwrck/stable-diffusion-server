from main import create_image_from_prompt, inpaint_image_from_prompt


def test_create_image_from_prompt_sync():
    imagebytesresult = create_image_from_prompt("a test prompt")
    assert imagebytesresult is not None

def test_inpaint_from_prompt_sync():
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    imagebytesresult = inpaint_image_from_prompt("a test prompt", img_url, mask_url, True)

    assert imagebytesresult is not None
