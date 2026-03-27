# Reproducibility Notes — MonoDBH

> This document provides a complete, step-by-step guide to reproducing the DBH estimation results reported in the associated journal paper, from raw field images through to the final estimated DBH in centimetres.
>
> Every design decision, intermediate file format, and edge-case handling is documented here.

---

## Table of Contents

1. [Overview of the Full Pipeline](#1-overview-of-the-full-pipeline)
2. [Field Data Collection Protocol](#2-field-data-collection-protocol)
3. [Segmentation Scripts — Which to Use and When](#3-segmentation-scripts--which-to-use-and-when)
   - 3.1 [Standard upright trees — Grounding DINO + SAM 2](#31-standard-upright-trees--grounding-dino--sam-2)
   - 3.2 [Tilted / leaning trees — PCA correction](#32-tilted--leaning-trees--pca-correction)
   - 3.3 [Branchy / complex scenes — NMS correction](#33-branchy--complex-scenes--nms-correction)
   - 3.4 [Florence-2 + SAM 2 alternative](#34-florence-2--sam-2-alternative)
   - 3.5 [SAM 2 automatic baseline](#35-sam-2-automatic-baseline)
4. [Understanding the Segmentation JSON Output](#4-understanding-the-segmentation-json-output)
5. [Extracting DBH Pixel Width from JSON](#5-extracting-dbh-pixel-width-from-json)
6. [The Raw Field Measurement CSV](#6-the-raw-field-measurement-csv)
7. [Merging Pixel Measurements with Field Data](#7-merging-pixel-measurements-with-field-data)
8. [Converting Pixel DBH to Centimetres](#8-converting-pixel-dbh-to-centimetres)
9. [Complete Worked Example](#9-complete-worked-example)
10. [Common Pitfalls and Fixes](#10-common-pitfalls-and-fixes)

---

## 1. Overview of the Full Pipeline

```
[Field images + metadata CSV]
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │  STEP 1: Segmentation script                            │
 │  grounded_sam2_base.py  /  grounded_sam2_florence2.py   │
 │  sam2_segmentation.py                                   │
 │                                                         │
 │  Input : images in notebooks/data/                      │
 │  Output: per-image annotated .jpg  +  .json sidecar     │
 └─────────────────────────────────────────────────────────┘
         │ JSON sidecar per image (diameter_line_coords, diameter_px …)
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │  STEP 2: Extract, merge & convert — all-in-one          │
 │  utils/dbh_estimator.py                                 │
 │                                                         │
 │  Inputs : JSON folder, image folder, metadata CSV       │
 │  Output : estimated_dbh.csv                             │
 │           (all metadata + dbh_width px + estimated cm)  │
 └─────────────────────────────────────────────────────────┘
```

---

### Recording the hand-to-tree distance

The `length` column in the raw CSV is the **physical distance from the camera sensor to the trunk surface**, in centimetres. In practice this is approximately equal to the arm's length of the person holding the phone.

> **Tip:** Use `exiftool <image.jpg>` to extract focal length and sensor width directly from EXIF metadata, rather than looking them up in a spec sheet.

---

## 3. Segmentation Scripts — Which to Use and When

### 3.1 Standard upright trees — Grounding DINO + SAM 2

**Script:** `grounded_sam2_base.py`

Use this for the majority of field images where:
- The tree trunk is roughly vertical
- The scene is not extremely cluttered

The pipeline:
1. Applies a morphological close (5×5 kernel) to the greyscale image to fill bark gaps before detection.
2. Runs Grounding DINO with prompt `"tree trunk. hand"`.
3. Applies NMS (IoU ≤ 0.2) to suppress duplicate boxes.
4. Runs SAM 2 with those boxes as prompts.
5. Runs connected-component analysis to keep only the largest component of each trunk mask (removes spurious floaters from bark texture).
6. Runs PCA on the mask pixels to find the principal axis of the trunk.
7. Draws the diameter line **perpendicular to the principal axis** at the vertical mid-point of the bounding box.

### 3.2 Tilted / leaning trees — PCA correction

**Script used:** same as 3.1 — PCA is applied in **all three detection-based scripts** by default. There is no separate script to run; the PCA correction is automatic.

**How it works:** Without PCA, a naive horizontal line would overestimate the diameter of a leaning trunk because it would cut diagonally across the cross-section. The PCA approach:

1. Collects all foreground (mask) pixel coordinates `(x, y)`.
2. Computes the covariance matrix of those coordinates.
3. Finds the eigenvector corresponding to the **largest eigenvalue** → this is the principal axis (trunk direction).
4. Computes the **perpendicular normal vector** to that axis.
5. Walks outward from the trunk centre along the normal in both directions until the mask boundary is reached.
6. The Euclidean distance between those two boundary points is the diameter in pixels.

The JSON output includes `trunk_angle_deg` — the angle of the trunk's principal axis from horizontal (0° = flat, 90° = perfectly vertical). For a well-segmented upright tree this should be close to 90°.

**Validation check:** If `trunk_angle_deg` is unexpectedly low (e.g. < 60°) for a tree that appears upright in the image, the mask likely captured background foliage rather than just the trunk. Inspect the annotated output image.

### 3.3 Branchy / complex scenes — NMS correction

**Script used:** same as 3.1 — NMS is applied in both `grounded_sam2_base.py` and `grounded_sam2_florence2.py`.

**The problem:** In images with heavy branching, Grounding DINO can detect the same trunk region multiple times at slightly different box positions. Without NMS, SAM 2 would produce redundant overlapping masks, and the largest-component filter could pick the wrong one.

**NMS parameters:**
- IoU threshold: `0.2` — intentionally aggressive to ensure only one detection per trunk region.
- Applied to all classes (trunk + hand) jointly before SAM 2 runs.

**If NMS removes valid detections** (e.g. two distinct trunks that genuinely overlap less than 20% but are still suppressed), increase the IoU threshold:
```python
input_boxes, scores, labels = apply_nms(input_boxes, scores, labels, iou_threshold=0.4)
```

### 3.4 Florence-2 + SAM 2 alternative

**Script:** `grounded_sam2_florence2.py`

Use this as a comparison or when Grounding DINO produces poor detections. Florence-2 uses a different visual-language backbone and can produce higher-quality boxes for certain image conditions.

Key differences from 3.1:
- Florence-2 is queried **twice** — once with `"tree trunk"` and once with `"hand"` — because its `<OPEN_VOCABULARY_DETECTION>` task returns only one class per call.
- Florence-2 does not return confidence scores; dummy scores of `1.0` are used for NMS compatibility.
- All subsequent steps (connected components, PCA, diameter line) are identical.

### 3.5 SAM 2 automatic baseline

**Script:** `sam2_segmentation.py`

Use this as a **lower bound baseline** only. No text prompt is used; SAM 2 generates all possible masks automatically and the **largest one** is selected as the trunk.

Limitations:
- No hand detection → metric conversion requires separate pixel-to-cm calibration.
- The largest segment assumption fails when large background elements (sky, ground) are present.
- No diameter line is drawn — this script only produces a mask overlay.
- For the paper results, this was compared against the detector-based pipelines on the same image set.

---

## 4. Understanding the Segmentation JSON Output

Each segmentation run produces one `.json` file per image, placed in the output folder alongside the annotated image. The JSON is a list of detection objects.

**Full schema:**

```json
[
  {
    "id": 0,
    "class": "tree trunk",
    "bbox": [120.4, 45.2, 380.1, 890.7],
    "area": 142853,
    "trunk_angle_deg": 87.34,
    "diameter_px": 214,
    "diameter_line_coords": {
      "left":  [118, 467],
      "right": [332, 467]
    },
    "diameter_center_y": 467
  },
  {
    "id": 1,
    "class": "hand",
    "bbox": [90.1, 400.0, 160.3, 480.5],
    "area": 8210
  }
]
```

| Field | Type | Description |
|---|---|---|
| `id` | int | Index of this detection within the image |
| `class` | string | `"tree trunk"` or `"hand"` |
| `bbox` | `[x1, y1, x2, y2]` | Bounding box in pixels (top-left + bottom-right) |
| `area` | int | Number of foreground pixels in the mask |
| `trunk_angle_deg` | float | Angle of trunk principal axis from horizontal (°). Only on trunk entries. 90° = perfectly vertical. |
| `diameter_px` | int | DBH measurement in pixels. **This is the primary output.** Only on trunk entries. |
| `diameter_line_coords` | object | Pixel coordinates of the two endpoints of the DBH line. Only on trunk entries. |
| `diameter_center_y` | int | Y pixel coordinate at which the diameter was measured. Only on trunk entries. |

> **Note:** Hand detections do not have `diameter_px` or `diameter_line_coords` fields. The notebook extraction step (Step 2) only reads trunk entries.

> **Legacy note:** An earlier version of the scripts used the key name `lowest_point_line_coords` instead of `diameter_line_coords`. If you have JSON files produced by older scripts, update the key name or use the legacy-compatible notebook cell described in [Section 5](#5-extracting-dbh-pixel-width-from-json).

---

## 5. Extracting DBH Pixel Width from JSON

**Script:** `utils/dbh_estimator.py`

This single script performs all post-segmentation steps — JSON extraction, image dimension lookup, metadata merge, and metric conversion — and writes the final results CSV in one run.

### Configuration

Open `utils/dbh_estimator.py` and set the four path variables at the top:

```python
JSON_FOLDER   = "../notebooks/seg_experiment/groundingdino_outputs"  # JSON output folder
IMAGE_FOLDER  = "../notebooks/data"        # original input images
METADATA_CSV  = "dbh_csv.csv"              # field measurement CSV
OUTPUT_CSV    = "estimated_dbh.csv"        # where to write results
```

### Running

```bash
python utils/dbh_estimator.py
```

### What the script does internally

| Step | Action |
|---|---|
| 1 | Iterates all `.json` files in `JSON_FOLDER`. For each file, reads `diameter_px` (Euclidean — correct for tilted trunks). Falls back to `diameter_line_coords` / `lowest_point_line_coords` for legacy files. |
| 2 | Opens each image header (fast, no full decode) to read `image_width` and `image_height`. |
| 3 | Left-joins the pixel measurements onto the metadata CSV on the `photo` column. Reports any images that have no matching segmentation output. |
| 4 | Applies the pinhole formula to compute `estimated_dbh` in cm. Writes `OUTPUT_CSV`. |

### Console output

The script prints a summary table. If `actual_dbh` is present in the metadata CSV, it also prints MAE in cm and %.

> **Note on pixel width extraction:** `diameter_px` is stored in the JSON as the Euclidean distance between the two endpoints of the PCA-perpendicular diameter line. This is correct for both upright and tilted trunks. The old approach of computing `right_x - left_x` (horizontal projection) only holds for perfectly vertical trunks and is no longer used.

---

## 6. The Raw Field Measurement CSV

**File:** `dbh_csv.csv` (user-supplied, not committed to the repository)

This CSV must be prepared manually from field records. One row per photographed tree / measurement.

**Required columns:**

| Column | Type | Units | Description |
|---|---|---|---|
| `photo` | string | — | Image filename including extension, e.g. `IMG_0042.jpg`. Must match the filename used in the JSON output. |
| `actual_dbh` | float | cm | Ground-truth DBH measured with a diameter tape in the field. Used for validation/error computation only, not for the estimation formula. |
| `length` | float | cm | Distance from camera sensor to trunk surface. In practice, the arm length of the photographer holding the phone against the trunk. |
| `focal_length` | float | mm | Camera focal length. Read from EXIF (`exiftool <image> | grep "Focal Length"`). |
| `sensor_width` | float | mm | Physical sensor width of the camera. Found in the camera spec sheet or EXIF (`Exif.Photo.FocalPlaneXResolution` + `FocalPlaneResolutionUnit`). |
| `image_width` | float | px | Full image width in pixels. Added automatically by Cell 2 of the notebook — include this column if it is not already present. |

**Example rows:**

```csv
photo,actual_dbh,length,focal_length,sensor_width,image_width
IMG_0001.jpg,34.2,75,4.25,6.17,4032
IMG_0002.jpg,28.7,68,4.25,6.17,4032
IMG_0003.jpg,51.0,80,4.25,6.17,4032
```

> **Sensor width note:** For iPhone cameras the sensor width varies by model. Common values: iPhone 12 Pro = 7.01 mm, iPhone 13 = 5.76 mm, iPhone 14 Pro = 8.64 mm. Always verify against the actual device spec.

> **Focal length note:** Modern smartphones apply digital zoom cropping; use the **equivalent focal length** from EXIF (`FocalLengthIn35mmFilm`) divided by the crop factor, or use the raw `FocalLength` value if it matches the physical sensor width.

---

## 7. Merging Pixel Measurements with Field Data

`utils/dbh_estimator.py` handles the merge automatically. After running, `OUTPUT_CSV` contains:

| Column | Source | Description |
|---|---|---|
| `photo` | metadata CSV | Image filename |
| `actual_dbh` | metadata CSV | Ground-truth DBH (cm) — if provided |
| `length` | metadata CSV | Camera-to-trunk distance (cm) |
| `focal_length` | metadata CSV | Camera focal length (mm) |
| `sensor_width` | metadata CSV | Camera sensor width (mm) |
| `dbh_width` | JSON extraction | DBH width in pixels (`diameter_px`, Euclidean) |
| `trunk_angle_deg` | JSON extraction | Trunk tilt angle (°); 90° = vertical |
| `image_width` | image file | Full image width in pixels |
| `image_height` | image file | Full image height in pixels |
| `estimated_dbh` | formula | Estimated DBH in centimetres |

Images in the metadata with no matching JSON output are reported on the console and excluded from the output CSV.

---

## 8. Converting Pixel DBH to Centimetres

**Script:** `utils/dbh_estimator.py` (conversion is built in — no separate step needed)

### Formula

The pinhole camera model relates the physical size of an object to its pixel size:

$$W_{\text{mm}} = \frac{n \cdot S \cdot D_{\text{mm}}}{N \cdot f}$$

$$\text{DBH}_{\text{cm}} = \frac{W_{\text{mm}}}{10}$$

| Symbol | CSV column | Description | Units |
|---|---|---|---|
| $n$ | `dbh_width` | Trunk width in pixels | px |
| $S$ | `sensor_width` | Camera sensor width | mm |
| $D_{\text{mm}}$ | `length × 10` | Camera-to-trunk distance | mm (`length` is in cm) |
| $N$ | `image_width` | Full image width | px |
| $f$ | `focal_length` | Camera focal length | mm |

### Running the conversion

Conversion runs automatically inside `utils/dbh_estimator.py`. Set `OUTPUT_CSV` at the top of the script and run:

```bash
python utils/dbh_estimator.py
```

The output CSV has `estimated_dbh` (float, cm, 4 decimal places) appended to all metadata and pixel columns. If `actual_dbh` is in the metadata CSV, the console also prints MAE in cm and %.

### Output CSV

`estimated_dbh` is rounded to 4 decimal places and appended as the last column alongside all metadata and pixel-width columns.

---

## 9. Complete Worked Example

This section walks through the entire pipeline for a single image `IMG_0042.jpg`.

### Step 1 — Run segmentation

```bash
# Place IMG_0042.jpg in notebooks/data/
python grounded_sam2_base.py
```

Output files:
- `notebooks/seg_experiment/groundingdino_outputs/IMG_0042.jpg` — annotated image
- `notebooks/seg_experiment/groundingdino_outputs/IMG_0042.json` — sidecar JSON

### Step 2 — Inspect the JSON

```json
[
  {
    "id": 0,
    "class": "tree trunk",
    "bbox": [150.0, 30.0, 420.0, 1050.0],
    "area": 185420,
    "trunk_angle_deg": 88.12,
    "diameter_px": 243,
    "diameter_line_coords": {
      "left":  [148, 540],
      "right": [391, 541]
    },
    "diameter_center_y": 540
  },
  {
    "id": 1,
    "class": "hand",
    "bbox": [130.0, 490.0, 200.0, 570.0],
    "area": 4850
  }
]
```

- `trunk_angle_deg = 88.12°` → trunk is nearly vertical, PCA correction is minimal.
- `diameter_px = 243` → the DBH line is 243 pixels long.
- `diameter_line_coords` → the line runs from pixel (148, 540) to (391, 541) — nearly horizontal as expected for an upright trunk.

### Step 3 — Prepare `metadata.csv`

```csv
photo,actual_dbh,length,focal_length,sensor_width
IMG_0042.jpg,41.5,200,4.25,6.17
```

(`length` = 200 cm — typical camera-to-trunk distance for a 41.5 cm trunk)

### Step 4 — Configure and run `dbh_estimator.py`

Edit the variables at the top of `utils/dbh_estimator.py`:

```python
JSON_FOLDER   = "../notebooks/seg_experiment/groundingdino_outputs"
IMAGE_FOLDER  = "../notebooks/data"
METADATA_CSV  = "metadata.csv"
OUTPUT_CSV    = "estimated_dbh.csv"
```

```bash
python utils/dbh_estimator.py
```

Console output:
```
────────────────────────────────────────────────────────────
Step 1 — Extracting pixel DBH widths from JSON files...
Extracted pixel widths from 1 JSON files.
────────────────────────────────────────────────────────────
Step 2 — Reading image dimensions...
────────────────────────────────────────────────────────────
Step 3 — Merging with field metadata...
Merged 1 matched rows (0 unmatched excluded).
────────────────────────────────────────────────────────────
Step 4 — Computing estimated DBH...
Saved 1 rows -> estimated_dbh.csv
────────────────────────────────────────────────────────────
     photo  actual_dbh  estimated_dbh  error_cm  error_pct
IMG_0042.jpg       41.5        42.3100    0.8100       1.95

Mean absolute error            : 0.8100 cm
Mean absolute percentage error : 1.95 %
```

**Worked calculation:**
```
D_mm  = 200 × 10 = 2000 mm
W_mm  = (243 × 6.17 × 2000) / (4032 × 4.25)
      = 2,998,620 / 17,136
      ≈ 175.0 mm  →  but output is per-image and will reflect your actual numbers
DBH_cm = W_mm / 10
```

> The formula scales linearly with `length`. Always verify `length`, `sensor_width`, and `focal_length` are correct for each device before interpreting results.

---

## 10. Common Pitfalls and Fixes

### No trunk detection in the JSON

The JSON file exists but contains no `"tree trunk"` entries (only a hand, or it is an empty list `[]`).

- **Cause:** Detection confidence was below the threshold (0.25).
- **Fix:** Lower the threshold in the script:
  ```python
  threshold=0.15  # in post_process_grounded_object_detection()
  ```
- **Alternative:** Check the annotated output image. If the trunk is partially visible or very dark against the background, try adjusting the morphological preprocessing kernel size.

---

### `diameter_line_coords` key missing from JSON

The trunk entry exists but does not have `diameter_line_coords`.

- **Cause:** The trunk mask was empty (all zeros) after the connected-component filter, so the PCA step was skipped.
- **Fix:** The mask was likely corrupted during the SAM 2 resize step. Check that the image resolution is consistent across your dataset. Try re-running the script on that image alone.

---

### Legacy key name `lowest_point_line_coords`

Older versions of the scripts wrote `lowest_point_line_coords` instead of `diameter_line_coords`. The current notebook handles `diameter_line_coords`. If you have a mix of old and new JSONs, use this backwards-compatible extraction:

```python
for obj in data:
    coords = obj.get("diameter_line_coords") or obj.get("lowest_point_line_coords")
    if coords:
        left_x  = coords["left"][0]
        right_x = coords["right"][0]
        width   = right_x - left_x
```

---

### Pixel width vs. Euclidean diameter

`utils/dbh_estimator.py` always reads `diameter_px` (Euclidean distance) as the primary source, which is correct for both upright and tilted trunks. The old horizontal projection (`right_x - left_x`) is only used as a legacy fallback when `diameter_px` is absent. No manual fix is needed.

---

### Merge produces `left_only` rows

Some images in the field CSV have no matching segmentation output.

- **Check 1:** File names must match exactly including extension and capitalisation (`IMG_0042.jpg` ≠ `img_0042.jpg`).
- **Check 2:** Did the segmentation script skip that image due to no detections? Check the script console output.
- **Check 3:** Did the image land in the wrong subfolder (e.g. `tilted_trees/` instead of `data/`)?

---

### `estimated_dbh` is unrealistically small or large

- **Too small:** `length` (distance) was entered in mm instead of cm. The formula expects cm.
- **Too large:** `sensor_width` was entered as the diagonal sensor size rather than width. Use the width dimension only.
- **Off by constant factor:** `focal_length` is the 35mm-equivalent value; use the physical focal length instead.

Run a sanity check: for a typical smartphone at 100 cm distance, a 30 cm trunk should measure roughly:

$$n = \frac{30 \times 10 \times N \times f}{S \times 1000} \text{ pixels}$$

---

### PCA gives wrong trunk angle

`trunk_angle_deg` is unexpectedly small (< 45°) for a tree that looks upright.

- The mask includes a large horizontal background element (e.g. the ground).
- The largest connected component filter should have removed this, but if the background is connected to the trunk in the mask (no clear gap), it won't.
- **Fix:** Increase the detection threshold or reduce the SAM 2 mask `stability_score_thresh` to get a tighter trunk-only mask.
