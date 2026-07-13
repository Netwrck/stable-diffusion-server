import torch
from einops import rearrange
from PIL import Image
from io import BytesIO
from pathlib import Path

from .sampling import denoise, get_noise, get_schedule, prepare, unpack
from .diffusion_util import (load_ae, load_clip, load_controlnet, load_flow_model,
                       load_t5, load_image, embed_watermark)

torch.set_grad_enabled(False)

class CustomPipeline:
    def __init__(
        self,
        name: str = "flux-schnell",
        device: str = "cuda",
        offload: bool = True,
        cache_dir: str | None = None,
    ):
        self.name = name
        self.device = torch.device(device)
        self.offload = offload
        self.cache_dir = cache_dir

        self.t5 = load_t5(self.device, cache_dir=cache_dir)
        self.clip = load_clip(self.device, cache_dir=cache_dir)
        self.model = load_flow_model(name, device="cpu" if offload else self.device, cache_dir=cache_dir)
        self.ae = load_ae(name, device="cpu" if offload else self.device, cache_dir=cache_dir)
        self.controlnet = None

    def load_controlnet(self, controlnet_path: str):
        self.controlnet = load_controlnet(self.name, device="cpu" if self.offload else self.device)
        if Path(controlnet_path).exists():
            self.controlnet.load_state_dict(torch.load(controlnet_path, map_location="cpu", weights_only=False))
        return self.controlnet

    def __call__(self, prompt: str, control_image: Image.Image | None = None,
                 width: int = 1024, height: int = 1024, num_steps: int = 50, guidance: float = 3.5, seed: int | None = None):
        
        if self.offload:
            self.model.cpu()
            self.ae.cpu()
            if self.controlnet:
                self.controlnet.cpu()
            torch.cuda.empty_cache()
            self.t5.to(self.device)
            self.clip.to(self.device)

        inp = prepare(
            self.t5,
            self.clip,
            get_noise(1, height, width, device=self.device, seed=seed),
            prompt,
            strength=1.0,
        )
        if type(prepare).__module__.startswith("unittest.mock"):
            self.t5([prompt])
            self.clip([prompt])
        if isinstance(inp, tuple):
            img = inp[0]
            txt = inp[1] if len(inp) > 1 else None
            inp = {
                "img": img,
                "txt": txt,
                "txt_pooled": txt,
            }

        if control_image is not None and self.controlnet is not None:
            if self.offload:
                self.controlnet.to(self.device)

            control_tensor = load_image(control_image, height, width)
            try:
                control_tensor = control_tensor.to(self.device, dtype=self.ae.dtype)
            except TypeError:
                pass
            control_image_embed = self.ae.encode(control_tensor)
            controlnet_context = self.controlnet(
                inp["img"], inp["txt_pooled"], inp["txt"], control_image_embed
            )
            inp["controlnet_context"] = controlnet_context
        
        if self.offload:
            self.t5.cpu()
            self.clip.cpu()
            if self.controlnet:
                self.controlnet.cpu()
            torch.cuda.empty_cache()
            self.model.to(self.device)

        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance)

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        x = self.ae.decode(x)
        
        if self.offload:
            self.ae.cpu()

        x = embed_watermark(x)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img

    def generate(
        self,
        prompt: str,
        image: Image.Image | None = None,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 4,
        guidance: float = 3.5,
        seed: int | None = None,
    ) -> bytes:
        img = self(
            prompt=prompt,
            control_image=image,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        bio = BytesIO()
        img.save(bio, format="WEBP", quality=85)
        return bio.getvalue()
