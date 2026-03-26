import torch
import numpy as np
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ── 1. Config ────────────────────────────────────────────────────────────────
CHECKPOINT    = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
INPUT_FOLDER  = "notebooks/data/"
OUTPUT_FOLDER = "notebooks/seg_experiment/sam2/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── 2. Load model ONCE ───────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reduce CUDA memory fragmentation (ignored on CPU)
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
else:
    print("⚠  No CUDA GPU detected — running on CPU. Inference will be significantly slower.")
sam2   = build_sam2(MODEL_CFG, CHECKPOINT, device=device)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=16,          # ← reduced from 32 (saves ~4x memory)
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=0,             # ← reduced from 1 (biggest memory saver)
    min_mask_region_area=500,
)

# ── 3. Supported extensions ───────────────────────────────────────────────────
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if os.path.splitext(f)[1].lower() in VALID_EXT
])
print(f"Found {len(image_files)} images in {INPUT_FOLDER}")

# ── 4. Loop ───────────────────────────────────────────────────────────────────
for idx, filename in enumerate(image_files):
    input_path  = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    print(f"[{idx+1}/{len(image_files)}] Processing: {filename}")

    # Load image
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f"  ⚠ Could not read {filename}, skipping.")
        continue

    # ── Optional: resize large images to save VRAM ───────────────────────────
    h, w = image_bgr.shape[:2]
    max_side = 1024
    if max(h, w) > max_side:
        scale     = max_side / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        print(f"  ↓ Resized to {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # bfloat16 autocast is supported on both CUDA and CPU; use float32 on CPU for safety
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            masks = mask_generator.generate(image_rgb)
    except (torch.OutOfMemoryError, MemoryError):
        print(f"  ⚠ OOM on {filename}, clearing cache and skipping.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

    if not masks:
        print(f"  ⚠ No masks found for {filename}, skipping.")
        continue

    # Pick largest mask
    largest    = sorted(masks, key=lambda m: m["area"], reverse=True)[0]
    trunk_mask = largest["segmentation"]
    print(f"  ✓ Masks: {len(masks)} | Largest area: {largest['area']} px²")

    # Green overlay
    trunk_vis = image_rgb.copy().astype(np.float32)
    trunk_vis[trunk_mask] = trunk_vis[trunk_mask] * 0.3 + np.array([0, 200, 80]) * 0.7
    trunk_vis = trunk_vis.astype(np.uint8)

    # Save
    output_bgr = cv2.cvtColor(trunk_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    print(f"  ✓ Saved → {output_path}")

    # ── Clear GPU cache after each image ─────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nDone! All images processed.")