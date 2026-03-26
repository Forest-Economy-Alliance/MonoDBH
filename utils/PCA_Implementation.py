import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
import json
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
 
# ------------------------------
# Step 1: Environment + Models
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
 
# Init SAM2 image predictor
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint).to(device)
image_predictor = SAM2ImagePredictor(sam2_image_model)
 
# Init Grounding DINO
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
 
# Input/Output folders
input_folder = "notebooks/tilted_trees"
output_folder = "notebooks/seg_experiment/tilted_trees_output"
os.makedirs(output_folder, exist_ok=True)
 
# Class prompt
text = "tree trunk trunk wooden trunk. hand"
 
# ------------------------------    
# Step 2: Process images
# ------------------------------
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
 
    image_path = os.path.join(input_folder, filename)
    image = Image.open(image_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    closed_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(closed_rgb)
 
    # --- Grounding DINO ---
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
 
    processed = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=0.25,
        target_sizes=[image.size[::-1]]
    )[0]
 
    input_boxes = processed["boxes"].cpu().numpy()
    labels = processed["labels"]
 
    if len(input_boxes) == 0:
        print(f"⚠️ No detections in {filename}")
        continue
 
    # --- Run SAM2 segmentation ---
    image_predictor.set_image(np.array(image))
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
 
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
 
    # Fix mask dimensions
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)
 
    original_h, original_w = image.size[1], image.size[0]
    fixed_masks = []
    for m in masks:
        if m.shape != (original_h, original_w):
            m_resized = cv2.resize(
                m.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            fixed_masks.append(m_resized)
        else:
            fixed_masks.append(m.astype(bool))
 
    fixed_masks = np.array(fixed_masks)
    cleaned_masks = []
    for mask, label in zip(fixed_masks, labels):
        if "trunk" in label.lower():
            # Convert mask to uint8
            binary_mask = mask.astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(binary_mask)

            # Keep only largest component
            max_area = 0
            largest_component = None
            for i in range(1, num_labels):  # skip background
                component = (labels_im == i)
                area = component.sum()
                if area > max_area:
                    max_area = area
                    largest_component = component

            cleaned_mask = np.zeros_like(binary_mask, dtype=bool)
            if largest_component is not None:
                cleaned_mask[largest_component] = True
            cleaned_masks.append(cleaned_mask)
        else:
            # Leave hand or other classes unchanged
            cleaned_masks.append(mask)

    fixed_masks = np.array(cleaned_masks)   
 
    # --- Build detections ---
    detections = sv.Detections(
    xyxy=input_boxes,
    mask=fixed_masks,
    class_id=np.arange(len(labels)),
    confidence=np.array(scores).reshape(-1) if scores is not None else None
)
 
 
    # --- Annotate ---
        # --- Annotate ---
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
 
    annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated = mask_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
 
    # --- Save JSON metadata ---
    results_json = []
    for i, (box, label, mask) in enumerate(zip(input_boxes, labels, fixed_masks)):
        area = int(mask.sum())
        results_json.append({
            "id": i,
            "class": label,
            "bbox": box.tolist(),
            "area": area
        })
 
    trunks = [r for r in results_json if "trunk" in r["class"].lower()]
    hands = [r for r in results_json if "hand" in r["class"].lower()]
    largest_trunk = max(trunks, key=lambda r: r["area"], default=None)
 
    filtered_results = []
    if largest_trunk:
        trunk_idx = largest_trunk["id"]
        trunk_box = largest_trunk["bbox"]
        trunk_mask = fixed_masks[trunk_idx]

        # --- PCA Orientation ---
        ys, xs = np.where(trunk_mask > 0)
        coords = np.column_stack((xs, ys))
        mean = np.mean(coords, axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_axis = eigvecs[:, np.argmax(eigvals)]

        # --- Angle with surface ---
        angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
        angle_deg = np.degrees(angle_rad)
        trunk_angle_with_surface = abs(angle_deg)
        if trunk_angle_with_surface > 90:
            trunk_angle_with_surface = 180 - trunk_angle_with_surface

        print(f"🌲 Corrected trunk angle with surface: {trunk_angle_with_surface:.2f}°")
        largest_trunk["trunk_angle_deg"] = round(float(trunk_angle_with_surface), 2)

        # --- PCA axis line (blue) ---
        length = 200
        x1_pca = int(mean[0] - principal_axis[0] * length)
        y1_pca = int(mean[1] - principal_axis[1] * length)
        x2_pca = int(mean[0] + principal_axis[0] * length)
        y2_pca = int(mean[1] + principal_axis[1] * length)
        cv2.line(annotated, (x1_pca, y1_pca), (x2_pca, y2_pca), (255, 0, 0), 2)
        cv2.circle(annotated, (int(mean[0]), int(mean[1])), 5, (0, 0, 255), -1)

        # --- Perpendicular diameter line ---
        normal_vec = np.array([-principal_axis[1], principal_axis[0]])
        normal_vec /= np.linalg.norm(normal_vec)

        x_min, y_min, x_max, y_max = map(int, trunk_box)
        y_center = int((y_min + y_max) / 2)

        # 🔸 Compute center X at that Y
        mask_row_indices = np.where((ys >= y_center - 5) & (ys <= y_center + 5))[0]
        if len(mask_row_indices) == 0:
            start_point = mean.copy()
        else:
            x_center = np.mean(xs[mask_row_indices])
            start_point = np.array([x_center, y_center], dtype=float)

        # Move outwards along normal
        step = 1
        max_steps = 1000
        point_left = start_point.copy()
        point_right = start_point.copy()

        for _ in range(max_steps):
            new_point = (int(point_left[0]), int(point_left[1]))
            if (0 <= new_point[0] < trunk_mask.shape[1] and
                0 <= new_point[1] < trunk_mask.shape[0] and
                trunk_mask[new_point[1], new_point[0]]):
                point_left -= normal_vec * step
            else:
                break

        for _ in range(max_steps):
            new_point = (int(point_right[0]), int(point_right[1]))
            if (0 <= new_point[0] < trunk_mask.shape[1] and
                0 <= new_point[1] < trunk_mask.shape[0] and
                trunk_mask[new_point[1], new_point[0]]):
                point_right += normal_vec * step
            else:
                break

        # Draw green diameter line
        p1 = (int(point_left[0]), int(point_left[1]))
        p2 = (int(point_right[0]), int(point_right[1]))
        cv2.line(annotated, p1, p2, (0, 255, 0), 2)

        # Save info
        diameter_px = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        largest_trunk["diameter_px"] = diameter_px
        largest_trunk["diameter_line_coords"] = {"left": p1, "right": p2}
        largest_trunk["diameter_center_y"] = int(y_center)

        filtered_results.append(largest_trunk)

 
 
    filtered_results.extend(hands)
 
    # Save image AFTER drawing line
    output_path = os.path.join(output_folder, f"pca_{filename}")
    cv2.imwrite(output_path, annotated)
    print(f"✅ Processed {filename} → {output_path}")
 
    # Save JSON
    json_path = os.path.join(output_folder, filename.rsplit(".", 1)[0] + ".json")
    with open(json_path, "w") as f:
        json.dump(filtered_results, f, indent=2)
 
    print(f"📑 Saved JSON metadata at {json_path}") 