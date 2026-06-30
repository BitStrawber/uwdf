# UWDF IP-Adapter Underwater Reference Path

This repository keeps the original CompVis Stable Diffusion code, and adds a
lightweight diffusers-based generation entry point for the UWDF underwater
synthesis route.

## Goal

Use three conditions for underwater ImageNet synthesis:

1. ImageNet source image -> VAE/img2img latent content condition.
2. Text prompt -> CLIP text condition.
3. Underwater reference image -> CLIP Vision + IP-Adapter image condition.

This implements the IP-Adapter part of "IP-Adapter: Text Compatible Image Prompt
Adapter for Text-to-Image Diffusion Models" through diffusers
`load_ip_adapter()` instead of vendoring the full IP-Adapter repository.

## Smoke Test

Activate an environment with modern diffusers IP-Adapter support, then run:

```bash
cd ~/xcx/exp_2/syn/stable-diffusion  # or this uwdf repo path

GPU=2 \
SOURCE_DIR=/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source/train \
REFERENCE_DIR=/media/HDD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi \
LIMIT=10 \
bash scripts/run_ipadapter_img2img_smoke.sh \
  2>&1 | tee logs/uwdf_ipadapter_smoke.log
```

Outputs are written by default to:

```text
/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ipadapter/smoke/generated
/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ipadapter/smoke/comparisons
```

The comparison images show source, underwater reference, and generated output.

## Full Generation

```bash
SPLIT=train \
GPU=2 \
LIMIT=0 \
bash scripts/run_ipadapter_img2img_generate.sh \
  2>&1 | tee logs/uwdf_ipadapter_train.log

SPLIT=val \
GPU=2 \
LIMIT=0 \
bash scripts/run_ipadapter_img2img_generate.sh \
  2>&1 | tee logs/uwdf_ipadapter_val.log
```

Default output root:

```text
/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ipadapter/{train,val}/generated
```

## Key Parameters

- `STRENGTH`: img2img denoising strength. Higher values change the source image
  more. Start with `0.35`.
- `GUIDANCE_SCALE`: text guidance strength. Start with `5.0`.
- `IP_ADAPTER_SCALE`: underwater reference strength. Start with `0.45`.
- `PROMPT`: default is `a realistic underwater photograph`.

A useful first grid is:

```text
STRENGTH: 0.25, 0.35, 0.45
IP_ADAPTER_SCALE: 0.25, 0.45, 0.65
GUIDANCE_SCALE: 5.0
```

If reference objects leak into generated images, lower `IP_ADAPTER_SCALE` and
add those objects to `NEGATIVE_PROMPT`.

## Weights

The script defaults to:

```text
Base model:      runwayml/stable-diffusion-v1-5
IP-Adapter repo: h94/IP-Adapter
Weight:          models/ip-adapter_sd15.bin
```

The first run downloads these through Hugging Face into `HF_HOME`, defaulting to
`/media/SSD1/huggingface`.
