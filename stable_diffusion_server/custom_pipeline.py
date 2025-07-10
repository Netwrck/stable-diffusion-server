import torch
from einops import rearrange
from PIL import Image

from .sampling import denoise, get_noise, get_schedule, prepare, unpack
from .diffusion_util import (load_ae, load_clip, load_controlnet, load_flow_model,
                       load_t5, load_image, embed_watermark)

torch.set_grad_enabled(False)

class CustomPipeline:
    def __init__(self, name: str = "flux-schnell", device: str = "cuda", offload: bool = True):
        self.name = name
        self.device = torch.device(device)
        self.offload = offload

        self.t5 = load_t5(self.device)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(name, device="cpu" if offload else self.device)
        self.ae = load_ae(name, device="cpu" if offload else self.device)
        self.controlnet = None

    def load_controlnet(self, controlnet_path: str):
        self.controlnet = load_controlnet(self.name, device="cpu" if self.offload else self.device)
        self.controlnet.load_state_dict(torch.load(controlnet_path, map_location="cpu"))

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

        if control_image is not None and self.controlnet is not None:
            if self.offload:
                self.controlnet.to(self.device)
            
            control_image_embed = self.ae.encode(load_image(control_image, height, width).to(self.device, dtype=self.ae.dtype))
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
