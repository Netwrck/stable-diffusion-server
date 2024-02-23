import gradio as gr
from PIL import Image

from main import create_image_from_prompt

# This demo needs to be run from the repo folder.
# python gradio_ui.py

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


with gr.Blocks() as _block:
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

interface_inputs = ["text", width, height]
demo = gr.Interface(fn=make_images, inputs=interface_inputs, outputs=gallery)

if __name__ == "__main__":
    demo.launch()
