#!/usr/bin/env python3
"""Score eval image sets: CLIP prompt adherence, plus PSNR/SSIM/LPIPS vs a reference set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from stable_diffusion_server.image_metrics import psnr, ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_images(directory: Path) -> tuple[list[Image.Image], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    images = [Image.open(r["path"]).convert("RGB") for r in manifest["records"]]
    return images, manifest


def clip_scores(images: list[Image.Image], prompts: list[str]) -> list[float]:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./models").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./models")
    scores = []
    with torch.inference_mode():
        for image, prompt in zip(images, prompts):
            inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            image_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_emb = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            score = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()
            scores.append(score)
    del model
    torch.cuda.empty_cache()
    return scores


def to_tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float()


def reference_metrics(images: list[Image.Image], ref_images: list[Image.Image]) -> dict:
    psnrs, ssims, lpips_scores = [], [], []
    lpips_model = None
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    except Exception:
        pass

    for img, ref in zip(images, ref_images):
        if img.size != ref.size:
            img = img.resize(ref.size)
        a, b = to_tensor(img), to_tensor(ref)
        psnrs.append(psnr(a, b))
        ssims.append(ssim(a, b))
        if lpips_model is not None:
            with torch.inference_mode():
                an = (a / 127.5 - 1.0).unsqueeze(0).to(DEVICE)
                bn = (b / 127.5 - 1.0).unsqueeze(0).to(DEVICE)
                lpips_scores.append(lpips_model(an, bn).item())

    result = {
        "psnr_vs_ref": sum(psnrs) / len(psnrs),
        "ssim_vs_ref": sum(ssims) / len(ssims),
    }
    if lpips_scores:
        result["lpips_vs_ref"] = sum(lpips_scores) / len(lpips_scores)
    return result


def contact_sheet(image_sets: dict[str, list[Image.Image]], out_path: Path, thumb: int = 256) -> None:
    names = list(image_sets)
    count = min(len(v) for v in image_sets.values())
    sheet = Image.new("RGB", (thumb * count, thumb * len(names)), "white")
    for row, name in enumerate(names):
        for col in range(count):
            sheet.paste(image_sets[name][col].resize((thumb, thumb)), (col * thumb, row * thumb))
    sheet.save(out_path)


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", required=True, help="eval output dirs (evals/out/<name>)")
    parser.add_argument("--ref", default="", help="reference dir for PSNR/SSIM/LPIPS")
    parser.add_argument("--output", default="evals/results/summary.json")
    parser.add_argument("--sheet", default="evals/results/contact_sheet.png")
    args = parser.parse_args()

    ref_images = None
    if args.ref:
        ref_images, _ = load_images(Path(args.ref))

    summary = {}
    image_sets = {}
    for directory in args.dirs:
        directory = Path(directory)
        images, manifest = load_images(directory)
        image_sets[manifest["name"]] = images
        prompts = [r["prompt"] for r in manifest["records"]]
        scores = clip_scores(images, prompts)
        entry = {
            "avg_latency_s": manifest["avg_latency_s"],
            "min_latency_s": manifest["min_latency_s"],
            "peak_vram_gb": manifest.get("peak_vram_gb"),
            "clip_score": sum(scores) / len(scores),
            "clip_min": min(scores),
        }
        if ref_images is not None and str(directory) != args.ref:
            entry.update(reference_metrics(images, ref_images))
        summary[manifest["name"]] = entry
        print(f"{manifest['name']}: {json.dumps({k: round(v, 4) for k, v in entry.items() if isinstance(v, float)})}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    sheet_path = Path(args.sheet)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(image_sets, sheet_path)
    print(f"Wrote {out} and {sheet_path}")


if __name__ == "__main__":
    main_cli()
