import os
import torch
from diffusers import FluxPipeline, FluxControlNetPipeline, ControlNetModel
from dfloat11 import DFloat11Model
from argparse import ArgumentParser


def is_dfloat11_available() -> bool:
    try:
        import dfloat11  # noqa: F401
        return True
    except Exception:
        return False

parser = ArgumentParser(
    description="Generate an image using FLUX with DFloat11 weights"
)
parser.add_argument(
    "--prompt",
    type=str,
    default="A futuristic cityscape at sunset, with flying cars, neon lights, and reflective water canals",
)
parser.add_argument("--save_path", type=str, default="image.png")
parser.add_argument(
    "--controlnet", action="store_true", help="Enable line controlnet LoRA"
)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="Number of inference steps",
)


def main() -> None:
    args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    # Load DFloat11 weights for the text transformer
    model_path = os.getenv("DF11_MODEL_PATH", "DFloat11/FLUX.1-dev-DF11")
    device = os.getenv("DF11_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    DFloat11Model.from_pretrained(
        model_path,
        device=device,
        bfloat16_model=pipe.transformer,
    )

    if args.controlnet:
        try:
            controlnet = ControlNetModel.from_pretrained(
                "black-forest-labs/flux-controlnet-canny",
                torch_dtype=torch.bfloat16,
            )
            cpipe = FluxControlNetPipeline(controlnet=controlnet, **pipe.components)
            cpipe.enable_model_cpu_offload()
            try:
                lora_path = os.getenv(
                    "CONTROLNET_LORA", "black-forest-labs/flux-controlnet-line-lora"
                )
                cpipe.load_lora_weights(lora_path, adapter_name="line")
                cpipe.set_adapters(["line"], adapter_weights=[1.0])
            except Exception:
                pass
            pipe = cpipe
        except Exception as e:
            print(f"Failed to load ControlNet: {e}")

    image = pipe(
        args.prompt,
        width=1920,
        height=1440,
        guidance_scale=3.5,
        num_inference_steps=args.steps,
        max_sequence_length=512,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]

    image.save(args.save_path)


if __name__ == "__main__":
    main()
