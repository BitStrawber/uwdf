#!/usr/bin/env python3
"""Stable Diffusion img2img + IP-Adapter underwater reference generation.

This is a lightweight UWDF entry point built on diffusers. It keeps ImageNet
content through img2img, uses text as the second condition, and uses an
underwater reference image through IP-Adapter as the third condition.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate underwater-style ImageNet images with SD img2img + IP-Adapter.")
    parser.add_argument("--source-dir", required=True,
                        help="ImageNet sampled source root, usually .../synthetic_imagenet/uwdf/source/train.")
    parser.add_argument("--reference-dir", required=True,
                        help="Underwater reference image directory. Files are searched recursively.")
    parser.add_argument("--out-dir", required=True,
                        help="Output root. Generated images preserve source relative class folders by default.")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--ip-adapter-repo", default="h94/IP-Adapter")
    parser.add_argument("--ip-adapter-subfolder", default="models")
    parser.add_argument("--ip-adapter-weight", default="ip-adapter_sd15.bin")
    parser.add_argument("--prompt", default="a realistic underwater photograph")
    parser.add_argument("--negative-prompt", default=(
        "cartoon, painting, illustration, deformed object, extra objects, "
        "fish, coral, diver, text, watermark, blurry, low quality, worst quality"))
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum source images to process. Use 0 for all images.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N source images after sorting.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.35,
                        help="Img2img denoising strength. Higher values change source content more.")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="Text classifier-free guidance scale.")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.45,
                        help="Reference-image condition strength.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "float32"), default="auto")
    parser.add_argument("--resize-mode", choices=("resize", "crop", "pad"), default="resize")
    parser.add_argument("--reference-mode", choices=("random", "round_robin"), default="random")
    parser.add_argument("--flat-output", action="store_true",
                        help="Do not preserve source class folders under generated/.")
    parser.add_argument("--save-comparison", action="store_true",
                        help="Save source/reference/generated side-by-side images under comparisons/.")
    parser.add_argument("--disable-safety-checker", action="store_true", default=True)
    return parser.parse_args()


def iter_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {s.lower() for s in IMAGE_SUFFIXES}
    )


def load_rgb(path: Path, size: Sequence[int], mode: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    width, height = size
    if mode == "resize":
        return img.resize((width, height), LANCZOS)
    if mode == "crop":
        return ImageOps.fit(img, (width, height), method=LANCZOS, centering=(0.5, 0.5))
    if mode == "pad":
        fitted = ImageOps.contain(img, (width, height), method=LANCZOS)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
        return canvas
    raise ValueError(mode)


def choose_reference(refs: Sequence[Path], index: int, rng: random.Random, mode: str) -> Path:
    if mode == "round_robin":
        return refs[index % len(refs)]
    return refs[rng.randrange(len(refs))]


def make_comparison(source: Image.Image, reference: Image.Image, generated: Image.Image) -> Image.Image:
    w, h = source.size
    label_h = 28
    canvas = Image.new("RGB", (w * 3, h + label_h), (255, 255, 255))
    canvas.paste(source, (0, label_h))
    canvas.paste(reference, (w, label_h))
    canvas.paste(generated, (w * 2, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "source", fill=(0, 0, 0))
    draw.text((w + 8, 8), "underwater reference", fill=(0, 0, 0))
    draw.text((w * 2 + 8, 8), "generated", fill=(0, 0, 0))
    return canvas


def relative_output_path(path: Path, source_root: Path, out_root: Path, flat: bool) -> Path:
    if flat:
        stem = "__".join(path.relative_to(source_root).with_suffix("").parts)
        return out_root / f"{stem}_ipadapter.png"
    rel = path.relative_to(source_root)
    return out_root / rel.parent / f"{rel.stem}_ipadapter.png"


def resolve_dtype(args: argparse.Namespace):
    if args.torch_dtype == "float16":
        return torch.float16
    if args.torch_dtype == "float32":
        return torch.float32
    return torch.float16 if args.device.startswith("cuda") and torch.cuda.is_available() else torch.float32


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dir).resolve()
    reference_root = Path(args.reference_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    generated_root = out_root / "generated"
    comparison_root = out_root / "comparisons"
    manifest_path = out_root / "manifest.jsonl"
    summary_path = out_root / "summary.json"
    out_root.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)
    if args.save_comparison:
        comparison_root.mkdir(parents=True, exist_ok=True)

    source_images = iter_images(source_root)
    references = iter_images(reference_root)
    if args.offset:
        source_images = source_images[args.offset:]
    if args.limit > 0:
        source_images = source_images[:args.limit]
    if not source_images:
        raise RuntimeError(f"No source images found under {source_root}")
    if not references:
        raise RuntimeError(f"No reference images found under {reference_root}")

    dtype = resolve_dtype(args)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    print("=========================================", flush=True)
    print("UWDF SD img2img + IP-Adapter", flush=True)
    print("=========================================", flush=True)
    print(f"source_root:       {source_root}", flush=True)
    print(f"reference_root:    {reference_root}", flush=True)
    print(f"out_root:          {out_root}", flush=True)
    print(f"source_images:     {len(source_images)}", flush=True)
    print(f"reference_images:  {len(references)}", flush=True)
    print(f"model_id:          {args.model_id}", flush=True)
    print(f"ip_adapter:        {args.ip_adapter_repo}/{args.ip_adapter_subfolder}/{args.ip_adapter_weight}", flush=True)
    print(f"device/dtype:      {device}/{dtype}", flush=True)
    print(f"strength:          {args.strength}", flush=True)
    print(f"guidance_scale:    {args.guidance_scale}", flush=True)
    print(f"ip_adapter_scale:  {args.ip_adapter_scale}", flush=True)
    print("=========================================", flush=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None if args.disable_safety_checker else None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        args.ip_adapter_repo,
        subfolder=args.ip_adapter_subfolder,
        weight_name=args.ip_adapter_weight,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    pipe = pipe.to(device)

    rng = random.Random(args.seed)
    records = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, source_path in enumerate(tqdm(source_images, desc="generate", unit="image")):
            ref_path = choose_reference(references, index, rng, args.reference_mode)
            init_image = load_rgb(source_path, (args.width, args.height), args.resize_mode)
            ref_image = load_rgb(ref_path, (args.width, args.height), args.resize_mode)
            generator = torch.Generator(device="cpu").manual_seed(args.seed + index)

            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=init_image,
                ip_adapter_image=ref_image,
                num_inference_steps=args.steps,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            out_path = relative_output_path(source_path, source_root, generated_root, args.flat_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(out_path)

            compare_path = None
            if args.save_comparison:
                compare_path = relative_output_path(source_path, source_root, comparison_root, True)
                compare_path.parent.mkdir(parents=True, exist_ok=True)
                make_comparison(init_image, ref_image, result).save(compare_path)

            record = {
                "index": index,
                "source": str(source_path),
                "reference": str(ref_path),
                "output": str(out_path),
                "comparison": str(compare_path) if compare_path else None,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "seed": args.seed + index,
                "strength": args.strength,
                "guidance_scale": args.guidance_scale,
                "ip_adapter_scale": args.ip_adapter_scale,
                "steps": args.steps,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    summary = {
        "source_root": str(source_root),
        "reference_root": str(reference_root),
        "out_root": str(out_root),
        "generated_root": str(generated_root),
        "comparison_root": str(comparison_root) if args.save_comparison else None,
        "manifest": str(manifest_path),
        "num_generated": len(records),
        "model_id": args.model_id,
        "ip_adapter_repo": args.ip_adapter_repo,
        "ip_adapter_subfolder": args.ip_adapter_subfolder,
        "ip_adapter_weight": args.ip_adapter_weight,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "ip_adapter_scale": args.ip_adapter_scale,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "resize_mode": args.resize_mode,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()


