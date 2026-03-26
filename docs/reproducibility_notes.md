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
[Field images + raw CSV]
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
 │  STEP 2: Extract pixel widths from JSON                 │
 │  utils/json_dbh-pixels_image-dimns_metadata.ipynb       │
 │  Cell 1 → output.csv          (name, width)             │
 │  Cell 2 → output_with_dims.csv (name, width,            │
 │            image_width, image_height)                   │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │  STEP 3: Merge with field measurement CSV               │
 │  utils/json_dbh-pixels_image-dimns_metadata.ipynb       │
 │  Cells 4-8 → final_output.csv                           │
 │  (pixel width + image dims + camera intrinsics +        │
 │   field-measured distance + actual DBH for validation)  │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │  STEP 4: Metric conversion                              │
 │  utils/dbh_metric_converter.py                          │
 │  Output: estimated_dbh (cm) per image                   │
 └─────────────────────────────────────────────────────────┘
```

---

## 2. Field Data Collection Protocol

Each photograph must satisfy the following conditions to be usable by the pipeline:

| Requirement | Reason |
|---|---|
| A human **hand is held flat against the trunk** at breast height (1.3 m) | The hand provides a physical scale reference visible in the image |
| The **entire trunk cross-section** at breast height is visible | The segmentation model needs to see both edges of the trunk |
| Camera is held **perpendicular to the trunk face** | Off-angle shots introduce foreshortening error in the pixel width |
| **Distance from phone to trunk** is recorded in centimetres | Required for the metric conversion formula (`length` column) |
| Camera **focal length and sensor width** are known | Required for the metric conversion formula; readable from EXIF |

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

**Notebook:** `utils/json_dbh-pixels_image-dimns_metadata.ipynb`

This notebook performs Steps 2 and 3 of the pipeline. Run all cells top to bottom after setting the path variables at the top of Cell 1.

### Cell 1 — Extract pixel widths from all JSON files → `output.csv`

**What it does:** Iterates all `.json` files in the output folder. For each trunk detection with a `diameter_line_coords` entry, computes:

```
width (px) = right_x - left_x
           = diameter_line_coords["right"][0] - diameter_line_coords["left"][0]
```

Note this uses only the **X coordinates** of the two endpoints. This is correct because the PCA normal vector is not guaranteed to be horizontal — for a tilted trunk the diameter line runs at an angle. The true pixel diameter is the Euclidean distance stored in `diameter_px`, **not** the X-only width. Use `diameter_px` directly if you want the correct oblique measurement.

> **Which field to use?**
>
> - Use `diameter_px` (from the JSON directly) for the most accurate pixel width — this is the Euclidean distance between the two endpoints of the PCA-perpendicular line.
> - The notebook Cell 1 currently computes `right_x - left_x` which is the **horizontal projection** of the diameter. For strictly vertical trunks these are the same. For tilted trunks, use `diameter_px` instead.

**Configure before running:**

```python
folder_path = "./output2"   # ← path to your JSON output folder
output_csv  = "output.csv"  # ← where to write results
```

**Output CSV schema:**

| Column | Description |
|---|---|
| `name` | Image filename (`.jpg`) |
| `width` | DBH width in pixels (horizontal projection of the diameter line) |

### Cell 2 — Append image dimensions → `output_with_dims.csv`

**What it does:** For each row in `output.csv`, opens the corresponding image file and appends its pixel dimensions. The `image_width` column is required by the metric conversion formula.

**Configure before running:**

```python
csv_path     = "output.csv"         # ← output from Cell 1
image_folder = "./data"             # ← folder containing your input images
output_csv   = "output_with_dims.csv"
```

**Output CSV schema:**

| Column | Description |
|---|---|
| `name` | Image filename |
| `width` | DBH width in pixels |
| `image_width` | Full image width in pixels |
| `image_height` | Full image height in pixels |

### Cell 3 — Import pandas

No configuration needed.

### Cell 4 — Load both CSVs

```python
output = pd.read_csv("output_with_dims.csv")  # pixel measurements
file   = pd.read_csv("dbh_csv.csv")           # raw field data
```

See [Section 6](#6-the-raw-field-measurement-csv) for the required schema of `dbh_csv.csv`.

### Cell 5 — Rename column for merge key

```python
output.rename(columns={"name": "photo"}, inplace=True)
```

Both DataFrames now share the key column `photo`.

### Cell 6 — Merge on filename

```python
merge = pd.merge(file, output, on="photo", how="left",
                 indicator=True, suffixes=('_left', '_right'), validate="1:1")
```

A left join is used so that field records without a matching JSON output (e.g. images where segmentation failed) are retained with `NaN` in the pixel columns. The `_merge` indicator lets you identify unmatched rows.

### Cell 7 — Inspect matched rows

```python
merge[merge['_merge'] == 'both']
```

Review this output. Any row with `_merge == 'left_only'` means the image was in the field CSV but the segmentation script produced no trunk detection. Check the annotated output image for that file.

### Cell 8 — Save final merged CSV

```python
merge = merge[merge['_merge'] == 'both']
merge.to_csv("final_output.csv", index=False)
```

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

After running all cells of the notebook you will have `final_output.csv` with the following combined schema:

| Column | Source | Description |
|---|---|---|
| `photo` | field CSV | Image filename |
| `actual_dbh` | field CSV | Ground-truth DBH (cm) |
| `length` | field CSV | Camera-to-trunk distance (cm) |
| `focal_length` | field CSV | Camera focal length (mm) |
| `sensor_width` | field CSV | Camera sensor width (mm) |
| `width` | JSON extraction | DBH width in pixels (from diameter_line_coords) |
| `image_width` | image file | Full image width in pixels |
| `image_height` | image file | Full image height in pixels |
| `_merge` | pandas merge | `"both"` = matched, `"left_only"` = no segmentation result |

Before passing to the metric converter, rename the `width` column to `dbh_width`:

```python
import pandas as pd
df = pd.read_csv("final_output.csv")
df.rename(columns={"width": "dbh_width"}, inplace=True)
df.to_csv("final_output_renamed.csv", index=False)
```

---

## 8. Converting Pixel DBH to Centimetres

**Function:** `utils/dbh_metric_converter.py` → `estimate_dbh(input_csv, output_csv)`

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

```python
from utils.dbh_metric_converter import estimate_dbh

df = estimate_dbh(
    input_csv="final_output_renamed.csv",
    output_csv="estimated_dbh_results.csv"
)

# Quick validation against ground truth
df["error_cm"]  = df["estimated_dbh"] - df["actual_dbh"]
df["error_pct"] = (df["error_cm"] / df["actual_dbh"]) * 100
print(df[["photo", "actual_dbh", "estimated_dbh", "error_cm", "error_pct"]].to_string())
```

### Output CSV

The function appends an `estimated_dbh` column (float, cm, rounded to 4 decimal places) to all existing columns and writes to the specified output path.

---

## 9. Complete Worked Example

This section walks through the entire pipeline for a single image `IMG_0042.jpg`.

### Step 1 — Run segmentation

```bash
# Place IMG_0042.jpg in notebooks/data/
python grounded_sam2_base.py
```

Output files:
- `notebooks/seg_experiment/groundingdino_outputs/pca_IMG_0042.jpg` — annotated image
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

### Step 3 — Run the notebook

Set in Cell 1:
```python
folder_path = "notebooks/seg_experiment/groundingdino_outputs"
output_csv  = "output.csv"
```

Set in Cell 2:
```python
image_folder = "notebooks/data"
```

Cell 1 produces `output.csv`:
```csv
name,width
IMG_0042.jpg,243
```

Cell 2 produces `output_with_dims.csv`:
```csv
name,width,image_width,image_height
IMG_0042.jpg,243,4032,3024
```

### Step 4 — Prepare `dbh_csv.csv`

```csv
photo,actual_dbh,length,focal_length,sensor_width
IMG_0042.jpg,41.5,72,4.25,6.17
```

### Step 5 — Run notebook Cells 3–8

After the merge, `final_output.csv` contains:
```csv
photo,actual_dbh,length,focal_length,sensor_width,width,image_width,image_height,_merge
IMG_0042.jpg,41.5,72,4.25,6.17,243,4032,3024,both
```

Rename `width` → `dbh_width` and run:

### Step 6 — Metric conversion

```python
from utils.dbh_metric_converter import estimate_dbh

df = estimate_dbh("final_output_renamed.csv", "results.csv")
# D_mm = 72 * 10 = 720 mm
# W_mm = (243 * 6.17 * 720) / (4032 * 4.25)
#       = 1,079,845.2 / 17,136
#       ≈ 63.01 mm
# DBH_cm = 63.01 / 10 ≈ 6.30 cm  ← if length was 72 cm
```

> In this worked example the numbers are illustrative. With a real 41.5 cm trunk the camera distance would typically be much larger (200–300 cm). The formula scales linearly with distance.

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

The notebook Cell 1 computes `right_x - left_x` (horizontal projection). For upright trunks this equals `diameter_px`. For a trunk tilted at angle θ from vertical, the horizontal projection underestimates the true diameter by a factor of `cos(90° - trunk_angle_deg)`. 

Use `diameter_px` directly from the JSON for tilted trees:

```python
for obj in data:
    if obj.get("class", "").lower() == "tree trunk" and "diameter_px" in obj:
        rows.append([file_name, obj["diameter_px"]])
```

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
