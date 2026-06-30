#!/usr/bin/env bash
set -euo pipefail

# Smoke test for UWDF Stable Diffusion img2img + IP-Adapter underwater reference.
# Run inside an environment with modern diffusers IP-Adapter support.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_DIR="${SOURCE_DIR:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source/train}"
REFERENCE_DIR="${REFERENCE_DIR:-/media/HDD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
IP_ADAPTER_REPO="${IP_ADAPTER_REPO:-h94/IP-Adapter}"
IP_ADAPTER_SUBFOLDER="${IP_ADAPTER_SUBFOLDER:-sdxl_models}"
IP_ADAPTER_WEIGHT="${IP_ADAPTER_WEIGHT:-ip-adapter_sdxl.bin}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-768}"
OUT_DIR="${OUT_DIR:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ipadapter/smoke}"
GPU="${GPU:-2}"
LIMIT="${LIMIT:-10}"
PROMPT="${PROMPT:-a realistic underwater photograph}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-cartoon, painting, illustration, deformed object, extra objects, fish, coral, diver, text, watermark, blurry, low quality, worst quality}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.35}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE:-0.45}"
SEED="${SEED:-2026}"
HF_HOME="${HF_HOME:-/media/SSD1/huggingface}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME

python scripts/ipadapter_img2img_underwater.py \
  --model-id "${MODEL_ID}" \
  --ip-adapter-repo "${IP_ADAPTER_REPO}" \
  --ip-adapter-subfolder "${IP_ADAPTER_SUBFOLDER}" \
  --ip-adapter-weight "${IP_ADAPTER_WEIGHT}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --source-dir "${SOURCE_DIR}" \
  --reference-dir "${REFERENCE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --prompt "${PROMPT}" \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --steps "${STEPS}" \
  --strength "${STRENGTH}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --ip-adapter-scale "${IP_ADAPTER_SCALE}" \
  --seed "${SEED}" \
  --save-comparison

