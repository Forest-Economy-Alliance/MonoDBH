import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
import json
import torchvision
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM

# ── 1. Config ─────────────────────────────────────────────────────────────────
device      = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Reduce CUDA memory fragmentation (ignored on CPU)
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if device == "cpu":
    print("⚠  No CUDA GPU detected — running on CPU. Inference will be significantly slower.")

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
FLORENCE2_ID    = "microsoft/Florence-2-large"

INPUT_FOLDER  = "notebooks/data"
OUTPUT_FOLDER = "notebooks/seg_experiment/florence_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Text prompts — Florence-2 open vocab detection
TRUNK_PROMPT = "tree trunk"
HAND_PROMPT  = "hand"

# ── 2. Load Models ONCE ───────────────────────────────────────────────────────
print("Loading Florence-2...")
florence2_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE2_ID, trust_remote_code=True, torch_dtype="auto"
).eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_ID, trust_remote_code=True)

print("Loading SAM2...")
sam2_model       = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
image_predictor  = SAM2ImagePredictor(sam2_model)

print("Models loaded ✓")

# ── 3. Florence-2 helper ──────────────────────────────────────────────────────
def run_florence2_ovd(image_pil, text_prompt):
    """Open Vocabulary Detection: returns bboxes + labels for a single prompt."""
    task   = "<OPEN_VOCABULARY_DETECTION>"
    prompt = task + text_prompt
    inputs = florence2_processor(
        text=prompt, images=image_pil, return_tensors="pt"
    ).to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = florence2_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    generated_text = florence2_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = florence2_processor.post_process_generation(
        generated_text, task=task, image_size=(image_pil.width, image_pil.height)
    )
    result = parsed[task]
    boxes  = np.array(result.get("bboxes", []))
    labels = result.get("bboxes_labels", [])
    return boxes, labels


# ── 4. NMS helper (same as your original) ────────────────────────────────────
def apply_nms(boxes, scores, labels, iou_threshold=0.2):
    if len(boxes) == 0:
        return np.array([]), np.array([]), []
    boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep     = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    return boxes_t[keep].numpy(), scores_t[keep].numpy(), [labels[i] for i in keep]


# ── 5. Main loop ──────────────────────────────────────────────────────────────
image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
print(f"Found {len(image_files)} images")

for filename in image_files:
    image_path = os.path.join(INPUT_FOLDER, filename)
    print(f"\nProcessing: {filename}")

    # Load image
    image     = Image.open(image_path).convert("RGB")
    img_cv    = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Morphological preprocessing (same as your original)
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    closed_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    image_proc = Image.fromarray(closed_rgb)

    # ── Florence-2 detection for trunk + hand separately ─────────────────────
    trunk_boxes, trunk_labels = run_florence2_ovd(image_proc, TRUNK_PROMPT)
    hand_boxes,  hand_labels  = run_florence2_ovd(image_proc, HAND_PROMPT)

    # Merge detections
    if len(trunk_boxes) > 0 and len(hand_boxes) > 0:
        input_boxes = np.vstack([trunk_boxes, hand_boxes])
        labels      = trunk_labels + hand_labels
    elif len(trunk_boxes) > 0:
        input_boxes = trunk_boxes
        labels      = trunk_labels
    elif len(hand_boxes) > 0:
        input_boxes = hand_boxes
        labels      = hand_labels
    else:
        print(f"  ⚠ No detections in {filename}, skipping.")
        continue

    # Dummy scores (Florence-2 OVD doesn't return scores)
    scores = np.ones(len(input_boxes))

    # NMS
    input_boxes, scores, labels = apply_nms(input_boxes, scores, labels, iou_threshold=0.2)
    print(f"  Detections after NMS: {len(input_boxes)} | labels: {labels}")

    if len(input_boxes) == 0:
        print(f"  ⚠ All boxes removed by NMS, skipping.")
        continue

    # ── SAM2 segmentation ─────────────────────────────────────────────────────
    # bfloat16 autocast is supported on both CUDA and CPU (PyTorch >= 1.10)
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        image_predictor.set_image(np.array(image_proc))
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            masks, sam_scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
    except (torch.OutOfMemoryError, MemoryError):
        print(f"  ⚠ OOM on {filename}, skipping.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

    # Fix mask dims
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    # Resize masks if needed
    original_h, original_w = image.size[1], image.size[0]
    fixed_masks = []
    for m in masks:
        if m.shape != (original_h, original_w):
            m = cv2.resize(
                m.astype(np.uint8), (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        fixed_masks.append(m.astype(bool))

    # Keep only largest connected component for trunk masks (same as your original)
    cleaned_masks = []
    for mask, label in zip(fixed_masks, labels):
        if "trunk" in label.lower():
            binary    = mask.astype(np.uint8)
            n, lbl_im = cv2.connectedComponents(binary)
            max_area, largest = 0, None
            for i in range(1, n):
                comp = (lbl_im == i)
                if comp.sum() > max_area:
                    max_area, largest = comp.sum(), comp
            clean = np.zeros_like(binary, dtype=bool)
            if largest is not None:
                clean[largest] = True
            cleaned_masks.append(clean)
        else:
            cleaned_masks.append(mask)
    fixed_masks = np.array(cleaned_masks)

    # ── Annotate ──────────────────────────────────────────────────────────────
    img_annotated = cv2.cvtColor(np.array(image_proc), cv2.COLOR_RGB2BGR)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=fixed_masks,
        class_id=np.arange(len(labels)),
        confidence=scores.reshape(-1),
    )
    img_annotated = sv.BoxAnnotator().annotate(scene=img_annotated.copy(), detections=detections)
    img_annotated = sv.MaskAnnotator().annotate(scene=img_annotated, detections=detections)
    img_annotated = sv.LabelAnnotator().annotate(img_annotated, detections=detections, labels=labels)

    # ── PCA + Diameter (identical to your original logic) ─────────────────────
    results_json = []
    for i, (box, label, mask) in enumerate(zip(input_boxes, labels, fixed_masks)):
        results_json.append({
            "id": i, "class": label,
            "bbox": box.tolist(), "area": int(mask.sum())
        })

    trunks = [r for r in results_json if "trunk" in r["class"].lower()]
    hands  = [r for r in results_json if "hand"  in r["class"].lower()]
    filtered_results = []

    for trunk in trunks:
        trunk_idx  = trunk["id"]
        trunk_box  = trunk["bbox"]
        trunk_mask = fixed_masks[trunk_idx].astype(bool)

        if trunk_mask.sum() == 0:
            continue

        ys, xs = np.where(trunk_mask > 0)
        if len(xs) < 2:
            continue

        # PCA
        coords   = np.column_stack((xs, ys))
        mean     = np.mean(coords, axis=0)
        centered = coords - mean
        cov      = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_axis   = eigvecs[:, np.argmax(eigvals)]

        # Trunk angle
        angle_deg = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
        trunk_angle = abs(angle_deg)
        if trunk_angle > 90:
            trunk_angle = 180 - trunk_angle
        trunk["trunk_angle_deg"] = round(float(trunk_angle), 2)

        # Draw PCA axis (blue)
        length = 200
        cv2.line(img_annotated,
                 (int(mean[0] - principal_axis[0]*length), int(mean[1] - principal_axis[1]*length)),
                 (int(mean[0] + principal_axis[0]*length), int(mean[1] + principal_axis[1]*length)),
                 (255, 0, 0), 2)
        cv2.circle(img_annotated, (int(mean[0]), int(mean[1])), 4, (0, 0, 255), -1)

        # Perpendicular diameter (green)
        normal_vec = np.array([-principal_axis[1], principal_axis[0]], dtype=float)
        normal_vec /= np.linalg.norm(normal_vec) + 1e-8

        x_min, y_min, x_max, y_max = map(int, trunk_box)
        y_center = int((y_min + y_max) / 2)

        mask_row = np.where((ys >= y_center - 5) & (ys <= y_center + 5))[0]
        start_x  = float(np.mean(xs[mask_row])) if len(mask_row) > 0 else mean[0]
        start_y  = float(y_center)
        start_pt = np.array([start_x, start_y], dtype=float)

        point_left  = start_pt.copy()
        point_right = start_pt.copy()
        for _ in range(2000):
            nx, ny = int(round(point_left[0])),  int(round(point_left[1]))
            if 0 <= nx < trunk_mask.shape[1] and 0 <= ny < trunk_mask.shape[0] and trunk_mask[ny, nx]:
                point_left -= normal_vec
            else:
                break
        for _ in range(2000):
            nx, ny = int(round(point_right[0])), int(round(point_right[1]))
            if 0 <= nx < trunk_mask.shape[1] and 0 <= ny < trunk_mask.shape[0] and trunk_mask[ny, nx]:
                point_right += normal_vec
            else:
                break

        p1 = (int(round(point_left[0])),  int(round(point_left[1])))
        p2 = (int(round(point_right[0])), int(round(point_right[1])))
        cv2.line(img_annotated, p1, p2, (0, 255, 0), 2)

        diameter_px = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        trunk["diameter_px"]           = diameter_px
        trunk["diameter_line_coords"]  = {"left": list(p1), "right": list(p2)}
        trunk["diameter_center_y"]     = int(y_center)
        filtered_results.append(trunk)
        print(f"  🌲 Trunk | angle={trunk_angle:.1f}° | diameter={diameter_px}px")

    filtered_results.extend(hands)

    # ── Save image + JSON ─────────────────────────────────────────────────────
    output_img_path  = os.path.join(OUTPUT_FOLDER, f"florence2_{filename}")
    output_json_path = os.path.join(OUTPUT_FOLDER, filename.rsplit(".", 1)[0] + ".json")

    cv2.imwrite(output_img_path, img_annotated)
    with open(output_json_path, "w") as f:
        json.dump(filtered_results, f, indent=2)

    print(f"  ✅ Saved → {output_img_path}")
    print(f"  📑 Saved → {output_json_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n✅ Done! All images processed.")