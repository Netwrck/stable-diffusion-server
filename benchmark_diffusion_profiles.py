#!/usr/bin/env python3
"""Benchmark Flux and Proteus/SDXL server inference profiles."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from PIL import Image

import main
from performance_optimizations import build_inference_kwargs


PROMPTS = [
    "a cinematic portrait of a friendly explorer, soft studio lighting, detailed",
    "a fantasy castle above a misty forest at sunrise, highly detailed",
    "a sleek retro futuristic robot reading in a quiet library",
]


def make_test_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (24, 32, 44))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                int(32 + 180 * x / max(width - 1, 1)),
                int(48 + 160 * y / max(height - 1, 1)),
                int(96 + 80 * ((x + y) / max(width + height - 2, 1))),
            )
    return image


def memory_snapshot() -> dict[str, float]:
    process = psutil.Process(os.getpid())
    snapshot = {
        "ram_gb": process.memory_info().rss / (1024**3),
    }
    if torch.cuda.is_available():
        snapshot.update(
            {
                "vram_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "vram_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "vram_peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
        )
    return snapshot


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_pipeline(model: str, task: str):
    if model == "flux":
        return main.get_img2img_pipe() if task == "style" else main.get_flux_pipe()
    if model in {"proteus", "sdxl"}:
        return main.get_sdxl_img2img_pipe() if task == "style" else main.get_sdxl_pipe()
    raise ValueError(f"Unsupported model: {model}")


def run_inference(
    pipe,
    model: str,
    task: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    family = "sdxl" if model in {"proteus", "sdxl"} else "flux"
    kwargs = build_inference_kwargs(family, "style" if task == "style" else "text", steps=steps)
    generator = torch.Generator("cpu").manual_seed(seed)
    input_image = make_test_image(width, height) if task == "style" else None

    reset_cuda_peak()
    before = memory_snapshot()
    start = time.perf_counter()
    with torch.inference_mode():
        if task == "style":
            result = pipe(
                prompt=prompt,
                image=input_image,
                strength=float(os.getenv("BENCH_STYLE_STRENGTH", "0.55")),
                generator=generator,
                **kwargs,
            ).images[0]
        else:
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                generator=generator,
                **kwargs,
            ).images[0]
    synchronize()
    latency_s = time.perf_counter() - start
    after = memory_snapshot()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

    return {
        "model": model,
        "task": task,
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps_requested": steps,
        "kwargs": kwargs,
        "latency_s": latency_s,
        "seed": seed,
        "memory_before": before,
        "memory_after": after,
        "output_path": str(output_path) if output_path else None,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for result in results:
        key = f"{result['model']}:{result['task']}:{result['steps_requested']}"
        bucket = summary.setdefault(key, {"count": 0, "latencies": [], "peak_vram": []})
        bucket["count"] += 1
        bucket["latencies"].append(result["latency_s"])
        bucket["peak_vram"].append(result["memory_after"].get("vram_peak_gb", 0.0))

    for bucket in summary.values():
        latencies = bucket.pop("latencies")
        peak_vram = bucket.pop("peak_vram")
        bucket["avg_latency_s"] = sum(latencies) / len(latencies)
        bucket["min_latency_s"] = min(latencies)
        bucket["max_latency_s"] = max(latencies)
        bucket["max_peak_vram_gb"] = max(peak_vram) if peak_vram else 0.0
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["flux", "proteus"], choices=["flux", "proteus", "sdxl"])
    parser.add_argument("--tasks", nargs="+", default=["text", "style"], choices=["text", "style"])
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 20])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", default="benchmark_results_diffusion.json")
    parser.add_argument("--image-dir", default="benchmark_outputs/diffusion_profiles")
    return parser.parse_args()


def main_cli() -> dict[str, Any]:
    args = parse_args()
    prompts = [args.prompt] if args.prompt else PROMPTS[: args.runs]
    while len(prompts) < args.runs:
        prompts.append(PROMPTS[len(prompts) % len(PROMPTS)])

    image_root = Path(args.image_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []

    for model in args.models:
        for task in args.tasks:
            pipe = get_pipeline(model, task)
            for steps in args.steps:
                for warmup_idx in range(args.warmup):
                    run_inference(
                        pipe,
                        model,
                        task,
                        prompts[0],
                        args.width,
                        args.height,
                        steps,
                        args.seed + warmup_idx,
                    )
                for run_idx, prompt in enumerate(prompts):
                    output_path = image_root / model / task / f"steps_{steps}_run_{run_idx}.webp"
                    result = run_inference(
                        pipe,
                        model,
                        task,
                        prompt,
                        args.width,
                        args.height,
                        steps,
                        args.seed + run_idx,
                        output_path,
                    )
                    print(
                        f"{model} {task} steps={steps} run={run_idx} "
                        f"{result['latency_s']:.3f}s peak_vram="
                        f"{result['memory_after'].get('vram_peak_gb', 0.0):.2f}GB"
                    )
                    results.append(result)

    payload = {
        "created_at": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "env": {
            key: os.getenv(key)
            for key in [
                "SDIF_OPTIMIZATION_PROFILE",
                "SDIF_CACHE_MODE",
                "SDIF_FIRST_BLOCK_CACHE_THRESHOLD",
                "SDIF_TORCH_COMPILE",
                "SDXL_USE_AYS",
                "PROTEUS_MODEL_REPO",
                "FLUX_MODEL_REPO",
            ]
        },
        "results": results,
        "summary": summarize(results),
    }

    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {output}")
    return payload


if __name__ == "__main__":
    main_cli()
