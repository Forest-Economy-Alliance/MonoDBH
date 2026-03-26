"""
dbh_estimator.py — Convert segmentation JSON outputs to estimated DBH (cm).

Usage
-----
    python dbh_estimator.py \\
        --json_folder  <path/to/segmentation/json/outputs> \\
        --image_folder <path/to/input/images> \\
        --metadata     <path/to/field_metadata.csv> \\
        --output       <path/to/estimated_dbh.csv>

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

import argparse
import json
import os
import sys

from PIL import Image
import pandas as pd


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
    """
    Left-join field metadata onto the pixel measurements on the ``photo`` column.
    Rows in the metadata that have no matching segmentation output are reported
    and excluded from the final result.
    """
    field_df = pd.read_csv(metadata_csv)

    required = {'photo', 'length', 'sensor_width', 'focal_length'}
    missing = required - set(field_df.columns)
    if missing:
        sys.exit(
            f'ERROR: metadata CSV is missing required column(s): {missing}\n'
            f'       Found columns: {list(field_df.columns)}'
        )

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
    return matched


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Convert pixel DBH to centimetres
# ──────────────────────────────────────────────────────────────────────────────

def estimate_dbh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pinhole camera model:

        W_mm = (dbh_width * sensor_width * D_mm) / (image_width * focal_length)
        DBH_cm = W_mm / 10

    where D_mm = length * 10  (length is stored in cm in the metadata CSV).
    """
    df = df.copy()
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
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate tree DBH (cm) from segmentation JSON outputs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--json_folder', required=True,
        help='Folder containing JSON sidecar files from the segmentation script.',
    )
    parser.add_argument(
        '--image_folder', required=True,
        help='Folder containing the original input images.',
    )
    parser.add_argument(
        '--metadata', required=True,
        help='CSV with per-image camera metadata (photo, length, sensor_width, focal_length).',
    )
    parser.add_argument(
        '--output', required=True,
        help='Path for the output CSV with estimated DBH.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.isdir(args.json_folder):
        sys.exit(f'ERROR: json_folder does not exist: {args.json_folder}')
    if not os.path.isdir(args.image_folder):
        sys.exit(f'ERROR: image_folder does not exist: {args.image_folder}')
    if not os.path.isfile(args.metadata):
        sys.exit(f'ERROR: metadata file does not exist: {args.metadata}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print('─' * 60)
    print('Step 1 — Extracting pixel DBH widths from JSON files...')
    pixel_df = extract_pixel_widths(args.json_folder)

    print('─' * 60)
    print('Step 2 — Reading image dimensions...')
    pixel_df = append_image_dimensions(pixel_df, args.image_folder)

    print('─' * 60)
    print('Step 3 — Merging with field metadata...')
    merged_df = merge_with_metadata(pixel_df, args.metadata)

    print('─' * 60)
    print('Step 4 — Computing estimated DBH...')
    result_df = estimate_dbh(merged_df)

    result_df.to_csv(args.output, index=False)
    print(f'Saved {len(result_df)} rows -> {args.output}')

    print('─' * 60)
    print_summary(result_df)
    print('─' * 60)


if __name__ == '__main__':
    main()
