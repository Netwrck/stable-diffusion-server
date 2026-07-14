#!/usr/bin/env python3
"""Generate a fixed prompt set with the env-configured Proteus pipeline for evals."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import main
from evals.prompts import EVAL_PROMPTS
from performance_optimizations import build_inference_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", default="text", choices=["text", "style"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--num-prompts", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--style-image", default="")
    parser.add_argument("--out-root", default="evals/out")
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "style":
        pipe = main.get_sdxl_img2img_pipe()
    else:
        pipe = main.get_sdxl_pipe()

    kwargs = build_inference_kwargs("sdxl", args.task, steps=args.steps, guidance_scale=args.guidance)

    style_image = None
    if args.task == "style":
        from PIL import Image

        if args.style_image:
            style_image = Image.open(args.style_image).convert("RGB").resize((args.width, args.height))
        else:
            style_image = Image.new("RGB", (args.width, args.height), (90, 110, 140))

    prompts = EVAL_PROMPTS[: args.num_prompts]

    def run(prompt: str, seed: int):
        generator = torch.Generator("cpu").manual_seed(seed)
        with torch.inference_mode():
            if args.task == "style":
                return pipe(prompt=prompt, image=style_image, strength=args.strength, generator=generator, **kwargs).images[0]
            return pipe(prompt=prompt, width=args.width, height=args.height, generator=generator, **kwargs).images[0]

    for i in range(args.warmup):
        run(prompts[0], args.seed + 1000 + i)
    torch.cuda.synchronize()

    records = []
    for idx, prompt in enumerate(prompts):
        start = time.perf_counter()
        image = run(prompt, args.seed)
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        path = out_dir / f"{idx:02d}.png"
        image.save(path)
        records.append({"index": idx, "prompt": prompt, "latency_s": latency, "path": str(path)})
        print(f"{args.name} [{idx}] {latency:.3f}s")

    latencies = [r["latency_s"] for r in records]
    manifest = {
        "name": args.name,
        "created_at": datetime.now().isoformat(),
        "task": args.task,
        "width": args.width,
        "height": args.height,
        "kwargs": {k: v for k, v in kwargs.items() if not hasattr(v, "shape")},
        "seed": args.seed,
        "env": {
            k: os.getenv(k)
            for k in [
                "SDIF_OPTIMIZATION_PROFILE", "SDIF_CACHE_MODE", "SDIF_DEEPCACHE_INTERVAL",
                "SDIF_TORCH_COMPILE", "SDIF_VAE", "SDIF_XFORMERS",
                "SDXL_SCHEDULER", "SDXL_SPEED_LORA", "SDXL_TIMESTEPS", "PROTEUS_MODEL_REPO",
            ]
        },
        "avg_latency_s": sum(latencies) / len(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "records": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{args.name}: avg {manifest['avg_latency_s']:.3f}s min {manifest['min_latency_s']:.3f}s")


if __name__ == "__main__":
    main_cli()
