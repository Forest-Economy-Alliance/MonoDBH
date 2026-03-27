"""
dbh_estimator.py — Convert segmentation JSON outputs to estimated DBH (cm).

Usage
-----
    Edit the path variables in the CONFIG section below, then run:

        python utils/dbh_estimator.py

Required columns in the metadata CSV
-------------------------------------
    photo         : image filename (e.g. "IMG_001.jpg")
    length        : camera-to-trunk distance in centimetres
    sensor_width  : camera sensor width in millimetres
    focal_length  : camera focal length in millimetres

Optional column
---------------
    actual_dbh    : ground-truth DBH in cm — enables MAE validation summary

Output CSV columns
------------------
    All metadata columns  +  dbh_width (px)  +  image_width (px)  +  estimated_dbh (cm)
    If actual_dbh is present: error_cm, error_pct are also appended.
"""

import json
import os
import sys

from PIL import Image
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths before running
# ──────────────────────────────────────────────────────────────────────────────

# Folder containing JSON sidecar files from the segmentation script.
# Use the output folder that matches the pipeline you ran:
#   Grounding DINO + SAM 2  ->  "../notebooks/seg_experiment/groundingdino_outputs"
#   Florence-2  + SAM 2     ->  "../notebooks/seg_experiment/florence_outputs"
#   SAM 2 automatic         ->  "../notebooks/seg_experiment/sam2"
JSON_FOLDER   = "../notebooks/seg_experiment/groundingdino_outputs"

# Folder containing the original input images (filenames must match JSON names).
IMAGE_FOLDER  = "../notebooks/data"

# CSV with per-image field measurements.
# Required columns: photo, length (cm), sensor_width (mm), focal_length (mm)
# Optional column : actual_dbh (cm) — enables MAE validation in the summary
METADATA_CSV  = "../notebooks/metadata.csv"

# Path for the output CSV with estimated DBH values.
OUTPUT_CSV    = "../notebooks/estimated_dbh.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract pixel DBH widths from JSON sidecar files
# ──────────────────────────────────────────────────────────────────────────────

def extract_pixel_widths(json_folder: str) -> pd.DataFrame:
    """
    Walk *json_folder* and pull the trunk pixel-width from each JSON file.

    Priority:
      1. ``diameter_px``  — Euclidean length of the PCA-perpendicular line.
         Correct for both upright and tilted trunks (current scripts).
      2. ``diameter_line_coords`` / ``lowest_point_line_coords`` — horizontal
         projection from saved endpoint coordinates (legacy files).

    Class matching mirrors the segmentation scripts: substring check on "trunk".
    """
    rows = []
    skipped = []

    for file_name in sorted(os.listdir(json_folder)):
        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(json_folder, file_name)
        image_name = file_name[:-5] + '.jpg'

        with open(file_path, 'r') as f:
            data = json.load(f)

        found = False
        for obj in data:
            if 'trunk' not in obj.get('class', '').lower():
                continue

            if 'diameter_px' in obj:
                width = obj['diameter_px']
            else:
                coords = (
                    obj.get('diameter_line_coords')
                    or obj.get('lowest_point_line_coords')
                )
                if coords is None:
                    continue
                # Horizontal projection — only accurate for vertical trunks
                width = coords['right'][0] - coords['left'][0]

            rows.append({
                'photo': image_name,
                'dbh_width': width,
                'trunk_angle_deg': obj.get('trunk_angle_deg', None),
            })
            found = True
            break  # one trunk per image

        if not found:
            skipped.append(image_name)

    if skipped:
        print(f'WARNING: {len(skipped)} JSON file(s) had no trunk detection:')
        for name in skipped:
            print(f'  {name}')

    df = pd.DataFrame(rows, columns=['photo', 'dbh_width', 'trunk_angle_deg'])
    print(f'Extracted pixel widths from {len(df)} JSON files.')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Append image pixel dimensions
# ──────────────────────────────────────────────────────────────────────────────

def append_image_dimensions(pixel_df: pd.DataFrame, image_folder: str) -> pd.DataFrame:
    """
    Add ``image_width`` and ``image_height`` columns by reading each image header.
    PIL only reads the header — it does not decode the full image — so this is fast.
    """
    widths, heights = [], []

    for photo in pixel_df['photo']:
        image_path = os.path.join(image_folder, photo)
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                w, h = img.size
        else:
            print(f'  WARNING: image not found: {image_path}')
            w, h = None, None
        widths.append(w)
        heights.append(h)

    pixel_df = pixel_df.copy()
    pixel_df['image_width'] = widths
    pixel_df['image_height'] = heights
    return pixel_df


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Merge with field metadata
# ──────────────────────────────────────────────────────────────────────────────

def merge_with_metadata(pixel_df: pd.DataFrame, metadata_csv: str) -> pd.DataFrame:
    field_df = pd.read_csv(metadata_csv)

    required = {'photo', 'length', 'sensor_width', 'focal_length'}
    missing = required - set(field_df.columns)
    if missing:
        sys.exit(
            f'ERROR: metadata CSV is missing required column(s): {missing}\n'
            f'       Found columns: {list(field_df.columns)}'
        )

    # Strip unit suffixes from columns that may contain strings like "5.49 mm"
    for col in ('focal_length', 'sensor_width'):
        if field_df[col].dtype == object:
            field_df[col] = (
                field_df[col]
                .astype(str)
                .str.replace(r'[^\d.]', '', regex=True)  # remove non-numeric chars
            )
            field_df[col] = pd.to_numeric(field_df[col], errors='coerce')
    merged = pd.merge(
        field_df,
        pixel_df,
        on='photo',
        how='left',
        indicator=True,
        suffixes=('_field', '_pixel'),
        validate='1:1',
    )

    unmatched = merged[merged['_merge'] == 'left_only']
    if len(unmatched) > 0:
        print(
            f'WARNING: {len(unmatched)} image(s) in metadata have no '
            'segmentation result and will be excluded:'
        )
        for name in unmatched['photo']:
            print(f'  {name}')

    matched = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])
    print(
        f'Merged {len(matched)} matched rows '
        f'({len(unmatched)} unmatched excluded).'
    )

    # Debug: Check for missing values in required columns
    print("DEBUG: Merged DataFrame:")
    print(matched.head())
    print("DEBUG: Missing values in required columns:")
    print(matched[['dbh_width', 'sensor_width', 'image_width', 'focal_length']].isnull().sum())

    return matched

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Convert pixel DBH to centimetres
# ──────────────────────────────────────────────────────────────────────────────

def estimate_dbh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pinhole camera model:

        W_mm = (dbh_width * sensor_width * D_mm) / (image_width * focal_length)
        DBH_cm = W_mm / 10

    where D_mm = length * 10  (length is stored in CENTIMETRES in the metadata CSV).
    """
    df = df.copy()

    numeric_columns = ['dbh_width', 'sensor_width', 'image_width', 'focal_length']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    invalid_rows = df[df[numeric_columns].isnull().any(axis=1)]
    if not invalid_rows.empty:
        print("WARNING: Rows with invalid or missing numeric values will be excluded:")
        print(invalid_rows[['photo'] + numeric_columns])
        df = df.dropna(subset=numeric_columns)

    D_mm = df['length'] * 10                                      # cm → mm
    W_mm = (
        df['dbh_width'] * df['sensor_width'] * D_mm
    ) / (df['image_width'] * df['focal_length'])
    df['estimated_dbh'] = (W_mm / 10).round(4)                   # mm → cm
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Summary / validation
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    if 'actual_dbh' in df.columns:
        df = df.copy()
        df['error_cm']  = (df['estimated_dbh'] - df['actual_dbh']).round(4)
        df['error_pct'] = ((df['error_cm'] / df['actual_dbh']) * 100).round(2)
        print(
            df[['photo', 'actual_dbh', 'estimated_dbh', 'error_cm', 'error_pct']]
            .to_string(index=False)
        )
        print(f'\nMean absolute error            : {df["error_cm"].abs().mean():.4f} cm')
        print(f'Mean absolute percentage error : {df["error_pct"].abs().mean():.2f} %')
    else:
        print(df[['photo', 'dbh_width', 'estimated_dbh']].to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Validate inputs
    if not os.path.isdir(JSON_FOLDER):
        sys.exit(f'ERROR: JSON_FOLDER does not exist: {JSON_FOLDER}')
    if not os.path.isdir(IMAGE_FOLDER):
        sys.exit(f'ERROR: IMAGE_FOLDER does not exist: {IMAGE_FOLDER}')
    if not os.path.isfile(METADATA_CSV):
        sys.exit(f'ERROR: METADATA_CSV does not exist: {METADATA_CSV}')

    output_dir = os.path.dirname(os.path.abspath(OUTPUT_CSV))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print('─' * 60)
    print('Step 1 — Extracting pixel DBH widths from JSON files...')
    pixel_df = extract_pixel_widths(JSON_FOLDER)

    print('─' * 60)
    print('Step 2 — Reading image dimensions...')
    pixel_df = append_image_dimensions(pixel_df, IMAGE_FOLDER)

    print('─' * 60)
    print('Step 3 — Merging with field metadata...')
    merged_df = merge_with_metadata(pixel_df, METADATA_CSV)

    print('─' * 60)
    print('Step 4 — Computing estimated DBH...')
    result_df = estimate_dbh(merged_df)

    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f'Saved {len(result_df)} rows -> {OUTPUT_CSV}')

    print('─' * 60)
    print_summary(result_df)
    print('─' * 60)


if __name__ == '__main__':
    main()
