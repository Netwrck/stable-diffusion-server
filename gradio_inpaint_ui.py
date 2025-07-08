import os
from io import BytesIO
from typing import Tuple

import gradio as gr
from PIL import Image

from main import create_image_from_prompt, inpaint_image_from_prompt


def _save_temp_image(img: Image.Image, name: str) -> str:
    path = f"/tmp/{name}.png"
    img.save(path)
    return path


def generate_or_inpaint(
    prompt: str,
    image_with_mask: Tuple[Image.Image, Image.Image] | None,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    negative_prompt: str,
):
    if image_with_mask is None:
        image_bytes = create_image_from_prompt(
            prompt, width, height, steps, {"guidance_scale": guidance_scale}
        )
        return Image.open(BytesIO(image_bytes))

    image, mask = image_with_mask
    if mask is None or mask.getbbox() is None:
        image_bytes = create_image_from_prompt(
            prompt, width, height, steps, {"guidance_scale": guidance_scale}
        )
        return Image.open(BytesIO(image_bytes))

    image_path = _save_temp_image(image, "in_base")
    mask_path = _save_temp_image(mask, "in_mask")

    image_bytes = inpaint_image_from_prompt(prompt, image_path, mask_path)
    os.remove(image_path)
    os.remove(mask_path)
    return Image.open(BytesIO(image_bytes))


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                width = gr.Number(label="Width", value=1024)
                height = gr.Number(label="Height", value=1024)
                steps = gr.Number(label="Steps", value=8)
                guidance_scale = gr.Slider(
                    minimum=0, maximum=20, value=1, label="Guidance Scale", step=0.1
                )
                image_editor = gr.Image(
                    label="Base Image", tool="editor", type="pil", source="upload"
                )
                run = gr.Button("Generate / Inpaint")
            output = gr.Image(label="Result")

        run.click(
            generate_or_inpaint,
            inputs=[
                prompt,
                image_editor,
                width,
                height,
                steps,
                guidance_scale,
                negative_prompt,
            ],
            outputs=output,
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
