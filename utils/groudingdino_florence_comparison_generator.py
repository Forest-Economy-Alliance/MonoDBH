import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import csv

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_FOLDER   = "notebooks/data/"
GDINO_JSON_DIR = "notebooks/seg_experiment/groundingdino_outputs/"        # ← your GDino JSON folder
FLORENCE_JSON_DIR = "notebooks/seg_experiment/florence_outputs/"  # ← your Florence JSON folder
OUTPUT_DIR     = "./notebooks/seg_experiment/groundingdino_florence_comparison_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_trunk(data):
    """Extract trunk entry from JSON list."""
    if data is None:
        return None
    for item in data:
        if "trunk" in item.get("class", "").lower():
            return item
    return None

def get_hand(data):
    if data is None:
        return None
    for item in data:
        if "hand" in item.get("class", "").lower():
            return item
    return None

def draw_bbox(img, bbox, color, label="", thickness=3):
    if bbox is None:
        return img
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return img

def draw_diameter_line(img, line_coords, color, thickness=3):
    if line_coords is None:
        return img
    # Support both key names
    left  = line_coords.get("left")
    right = line_coords.get("right")
    if left and right:
        cv2.line(img, tuple(map(int, left)), tuple(map(int, right)), color, thickness)
        mid_x = int((left[0] + right[0]) / 2)
        mid_y = int((left[1] + right[1]) / 2)
        px = int(np.linalg.norm(np.array(right) - np.array(left)))
        cv2.putText(img, f"{px}px", (mid_x, mid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return img

def compute_bbox_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return round(inter/union, 4) if union > 0 else 0.0

def box_area(bbox):
    if bbox is None: return 0
    return int((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))

def get_diameter(trunk):
    """Get diameter px — supports both key names."""
    if trunk is None: return 0
    return trunk.get("diameter_px") or trunk.get("lowest_point_width_px") or 0

def get_diameter_coords(trunk):
    if trunk is None: return None
    return trunk.get("diameter_line_coords") or trunk.get("lowest_point_line_coords")

# ── Main loop ─────────────────────────────────────────────────────────────────
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
print(f"Found {len(image_files)} images\n")

all_metrics = []

for filename in image_files:
    stem = filename.rsplit(".", 1)[0]

    # Load JSONs
    gdino_data   = load_json(os.path.join(GDINO_JSON_DIR,    stem + ".json"))
    florence_data= load_json(os.path.join(FLORENCE_JSON_DIR, stem + ".json"))

    if gdino_data is None and florence_data is None:
        print(f"⚠ No JSON found for {filename}, skipping.")
        continue

    gdino_trunk   = get_trunk(gdino_data)
    florence_trunk= get_trunk(florence_data)
    gdino_hand    = get_hand(gdino_data)
    florence_hand = get_hand(florence_data)

    # Load image
    image_path = os.path.join(IMAGE_FOLDER, filename)
    img_orig   = cv2.imread(image_path)
    if img_orig is None:
        print(f"⚠ Could not read image {filename}, skipping.")
        continue

    h, w = img_orig.shape[:2]

    # ── Build 3 panels: Original | GDino | Florence-2 ─────────────────────────
    img_gdino    = img_orig.copy()
    img_florence = img_orig.copy()

    # GDino — draw trunk bbox (blue) + diameter (green) + hand bbox (red)
    if gdino_trunk:
        draw_bbox(img_gdino, gdino_trunk["bbox"], (255, 80, 0), "GDino Trunk")
        draw_diameter_line(img_gdino, get_diameter_coords(gdino_trunk), (0, 255, 0))
    if gdino_hand:
        draw_bbox(img_gdino, gdino_hand["bbox"], (0, 0, 255), "Hand")

    # Florence-2 — draw trunk bbox (orange) + diameter (green) + hand bbox (red)
    if florence_trunk:
        draw_bbox(img_florence, florence_trunk["bbox"], (0, 120, 255), "Florence Trunk")
        draw_diameter_line(img_florence, get_diameter_coords(florence_trunk), (0, 255, 0))
    if florence_hand:
        draw_bbox(img_florence, florence_hand["bbox"], (0, 0, 255), "Hand")

    # ── Metrics ───────────────────────────────────────────────────────────────
    bbox_iou = compute_bbox_iou(
        gdino_trunk["bbox"]    if gdino_trunk    else None,
        florence_trunk["bbox"] if florence_trunk else None
    )
    gdino_diam   = get_diameter(gdino_trunk)
    florence_diam= get_diameter(florence_trunk)
    diam_diff    = abs(gdino_diam - florence_diam)
    diam_diff_pct= round(diam_diff / gdino_diam * 100, 2) if gdino_diam > 0 else 0

    metrics = {
        "filename"            : filename,
        "gdino_trunk_area"    : gdino_trunk["area"]    if gdino_trunk    else 0,
        "florence_trunk_area" : florence_trunk["area"] if florence_trunk else 0,
        "gdino_bbox_area"     : box_area(gdino_trunk["bbox"]    if gdino_trunk    else None),
        "florence_bbox_area"  : box_area(florence_trunk["bbox"] if florence_trunk else None),
        "bbox_iou"            : bbox_iou,
        "gdino_diameter_px"   : gdino_diam,
        "florence_diameter_px": florence_diam,
        "diameter_diff_px"    : diam_diff,
        "diameter_diff_pct"   : diam_diff_pct,
        "gdino_angle_deg"     : gdino_trunk.get("trunk_angle_deg", "N/A")    if gdino_trunk    else "N/A",
        "florence_angle_deg"  : florence_trunk.get("trunk_angle_deg", "N/A") if florence_trunk else "N/A",
    }
    all_metrics.append(metrics)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"GDino vs Florence-2  |  {filename}",
                 fontsize=13, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[5, 1.2], hspace=0.35)

    panels = [
        ("Original",          img_orig),
        ("Grounding DINO",    img_gdino),
        ("Florence-2",        img_florence),
    ]
    for i, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # ── Metrics table ─────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.axis("off")

    col_labels = ["Metric", "Grounding DINO", "Florence-2", "Difference", "Diff %"]
    table_data = [
        ["Trunk mask area (px²)",
         f"{metrics['gdino_trunk_area']:,}",
         f"{metrics['florence_trunk_area']:,}",
         f"{metrics['gdino_trunk_area'] - metrics['florence_trunk_area']:,}",
         f"{abs(metrics['gdino_trunk_area']-metrics['florence_trunk_area']) / max(metrics['gdino_trunk_area'],1) * 100:.1f}%"],

        ["BBox area (px²)",
         f"{metrics['gdino_bbox_area']:,}",
         f"{metrics['florence_bbox_area']:,}",
         f"{metrics['gdino_bbox_area'] - metrics['florence_bbox_area']:,}",
         f"{abs(metrics['gdino_bbox_area']-metrics['florence_bbox_area']) / max(metrics['gdino_bbox_area'],1) * 100:.1f}%"],

        ["BBox IoU",
         f"{bbox_iou:.4f}", "—", "—", "—"],

        ["Diameter (px)",
         str(gdino_diam),
         str(florence_diam),
         str(diam_diff),
         f"{diam_diff_pct}%"],

        ["Trunk angle (°)",
         str(metrics['gdino_angle_deg']),
         str(metrics['florence_angle_deg']),
         str(abs(float(metrics['gdino_angle_deg']) - float(metrics['florence_angle_deg']))
             if metrics['gdino_angle_deg'] != "N/A" and metrics['florence_angle_deg'] != "N/A"
             else "N/A"),
         "—"],
    ]

    tbl = ax_t.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2C3E50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Highlight diff columns
    for row in range(1, len(table_data)+1):
        tbl[(row, 3)].set_facecolor("#FFF3CD")
        tbl[(row, 4)].set_facecolor("#FFE0E0")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"compare_{filename}")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {filename} | IoU={bbox_iou:.3f} | Diam GDino={gdino_diam}px Florence={florence_diam}px | Δ={diam_diff}px ({diam_diff_pct}%)")

# ── Summary CSV ───────────────────────────────────────────────────────────────
if all_metrics:
    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)

    # Aggregate stats
    valid_ious   = [m["bbox_iou"]         for m in all_metrics if m["bbox_iou"] > 0]
    valid_diams  = [m["diameter_diff_px"] for m in all_metrics if m["gdino_diameter_px"] > 0]
    valid_pcts   = [m["diameter_diff_pct"]for m in all_metrics if m["gdino_diameter_px"] > 0]

    print(f"\n{'='*55}")
    print(f"  Total images compared     : {len(all_metrics)}")
    print(f"  Mean BBox IoU             : {np.mean(valid_ious):.3f} ± {np.std(valid_ious):.3f}")
    print(f"  Mean diameter diff        : {np.mean(valid_diams):.1f} ± {np.std(valid_diams):.1f} px")
    print(f"  Mean diameter diff %      : {np.mean(valid_pcts):.2f}%")
    print(f"  📊 CSV saved → {csv_path}")
    print(f"{'='*55}")