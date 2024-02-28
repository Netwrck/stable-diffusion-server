import gradio as gr
from PIL import Image

from main import create_image_from_prompt, style_transfer_image_from_prompt
from PIL import Image

# This demo needs to be run from the repo folder.
# python gradio_ui.py

def style_transfer_make_images(image_array, text, strength):
    image = Image.fromarray(image_array)
    augmentations = [
        "",
        "awesome",
        "amazing",
        # "beautiful",
        # "hd",
        # "4k",
        "detailed aesthetic",
        "wonderful",
        "anime hd",
    ]
    images = []
    for aug in augmentations:
        print(aug)
        prompt = text + " " + aug
        bio = style_transfer_image_from_prompt(prompt, None, strength, False, image)
        save_name  = prompt.replace(" ", "-")
        save_path = f"outputs/{save_name}.webp"
        with open(save_path, "wb") as f:
            f.write(bio)

        pil_image = Image.open(save_path)

        images.append(pil_image)
        yield images
    return images


def make_images(text, width, height):
    augmentations = [
        "",
        "awesome",
        "amazing",
        # "beautiful",
        # "hd",
        # "4k",
        "detailed aesthetic",
        "wonderful",
        "anime hd",
    ]
    images = []
    for aug in augmentations:
        print(aug)
        prompt = text + " " + aug
        bio = create_image_from_prompt(prompt, width, height)
        save_name  = prompt.replace(" ", "-")
        save_path = f"outputs/{save_name}.webp"
        with open(save_path, "wb") as f:
            f.write(bio)

        pil_image = Image.open(save_path)

        images.append(pil_image)
        yield images
    return images

gallery = gr.Gallery(
        label="Generated images",
        show_label=False,
        elem_id="gallery",
        columns=[4],
        rows=[4],
        object_fit="contain",
        height="auto",
    )

width = gr.Number(value=1024, label="Width")
height = gr.Number(value=1024, label="Height")
strength = gr.Number(value=0.65, label="Strength", step=0.01, minimum=0, maximum=1)
prompt = gr.Textbox(lines=3, label="Prompt", placeholder="Anime detailed hd aesthetic best quality")
image_input = gr.Image(label="Image")

with gr.Blocks() as _block:
    
    interface_inputs = ["text", width, height]
    with gr.Tab("Text To Image"):
        demo = gr.Interface(fn=make_images, inputs=interface_inputs, outputs=gallery)
        
    with gr.Tab("Style Transfer"):
        style_transfer_interface_inputs = [image_input, prompt, strength]
        demo_style_transfer = gr.Interface(fn=style_transfer_make_images, inputs=style_transfer_interface_inputs, outputs=gallery)
    if __name__ == "__main__":
        # demo.launch()
        _block.launch()
