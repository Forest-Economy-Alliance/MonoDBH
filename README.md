# MonoDBH: Monocular Tree DBH Estimation via Segmentation and Scale Reference

> **Associated with a journal paper submission.** This repository contains all code, utilities, and instructions required to fully reproduce the results.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
  - [Approach 1 — Grounding DINO + SAM 2](#approach-1--grounding-dino--sam-2)
  - [Approach 2 — Florence-2 + SAM 2](#approach-2--florence-2--sam-2)
  - [Approach 3 — SAM 2 Automatic (no detector)](#approach-3--sam-2-automatic-no-detector)
- [DBH Metric Conversion](#dbh-metric-conversion)
- [Utility Modules](#utility-modules)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Virtual Environment — DBH Estimator Only](#virtual-environment--dbh-estimator-only)
  - [Prerequisites](#prerequisites) *(manual / conda)*
  - [Step-by-step Setup](#step-by-step-setup)
  - [Download Checkpoints](#download-checkpoints)
- [Usage](#usage)
  - [Running Grounding DINO + SAM 2](#running-grounding-dino--sam-2)
  - [Running Florence-2 + SAM 2](#running-florence-2--sam-2)
  - [Running SAM 2 Automatic](#running-sam-2-automatic)
  - [Converting Pixel DBH to Centimetres](#converting-pixel-dbh-to-centimetres)
- [Data Collection Procedure](#data-collection-procedure)
- [Data and Folder Structure](#data-and-folder-structure)
- [Output Format](#output-format)
- [Reproducibility Notes](#reproducibility-notes) — see also [docs/reproducibility_notes.md](docs/reproducibility_notes.md)
- [Third-Party Credits and Licences](#third-party-credits-and-licences)
- [Licence](#licence)
- [Code of Conduct](#code-of-conduct)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

**MonoDBH** estimates the **Diameter at Breast Height (DBH)** of a tree from a single (monocular) RGB image. A human hand is placed against the trunk at breast height to serve as a physical scale reference. The pipeline:

1. Detects and segments the tree trunk and the reference hand using a combination of open-vocabulary detection models and SAM 2.
2. Finds the trunk's primary orientation via **Principal Component Analysis (PCA)** to correctly handle tilted and leaning trees.
3. Draws a diameter line **perpendicular to the trunk axis** at mid-height and measures it in pixels.
4. Converts the pixel measurement to centimetres using camera intrinsics and the known subject distance.

Three detection back-ends are provided and compared:

| Script | Detector | Segmentor | Notes |
|---|---|---|---|
| `grounded_sam2_base.py` | Grounding DINO | SAM 2 | Primary pipeline |
| `grounded_sam2_florence2.py` | Florence-2 | SAM 2 | Alternative pipeline |
| `sam2_segmentation.py` | — (automatic) | SAM 2 | Baseline: largest segment = trunk |

---

## Pipeline Architecture

### Approach 1 — Grounding DINO + SAM 2

**Script:** `grounded_sam2_base.py`

```
Input image
    │
    ▼
Morphological close (5×5 kernel, grayscale)   ← gap-fill preprocessing
    │
    ▼
Grounding DINO  ──  prompt: "tree trunk. hand"
    │  bounding boxes + scores
    ▼
NMS (IoU ≤ 0.2)   ← removes overlapping duplicate detections
    │  filtered boxes
    ▼
SAM 2 Image Predictor  ──  box-prompted segmentation
    │  binary masks
    ▼
Connected-component filter   ← keeps only the largest component per trunk mask
    │
    ▼
PCA on trunk mask pixels
    │  principal axis vector
    ▼
Perpendicular diameter line   ← walks outward from centre along normal vector
    │  diameter_px
    ▼
JSON + annotated image output
```

**Key parameters:**
- Detection threshold: `0.25`
- NMS IoU threshold: `0.2`
- SAM 2 checkpoint: `sam2.1_hiera_large`
- Model config: `configs/sam2.1/sam2.1_hiera_l.yaml`

---

### Approach 2 — Florence-2 + SAM 2

**Script:** `grounded_sam2_florence2.py`

Identical architecture to Approach 1, with the following differences:

- **Detector:** Florence-2-large (`microsoft/Florence-2-large`) using the `<OPEN_VOCABULARY_DETECTION>` task.
- **Dual prompting:** trunk and hand are queried with **separate prompts** (`"tree trunk"`, `"hand"`) and results merged before NMS, because Florence-2 returns one object class per call.
- Florence-2 does not return confidence scores; dummy scores of `1.0` are assigned for NMS compatibility.
- Runs with `torch.bfloat16` autocast on CUDA.

---

### Approach 3 — SAM 2 Automatic (no detector)

**Script:** `sam2_segmentation.py`

- Uses `SAM2AutomaticMaskGenerator` — no text prompt, no bounding boxes.
- **Assumption:** for this dataset the largest segment in the image is the tree trunk.
- Reduced `points_per_side=16` and `crop_n_layers=0` to lower GPU memory usage.
- Outputs a green overlay on the selected trunk mask. DBH line measurement is not applied in this baseline.

---

## DBH Metric Conversion

**Script:** `utils/dbh_estimator.py`

After the segmentation scripts produce per-image JSON sidecar files, run `dbh_estimator.py` to extract pixel widths, read image dimensions, merge with your field metadata, and compute real-world DBH in centimetres — all in one step.

Real-world DBH is computed using the pinhole camera model:

$$W_{\text{mm}} = \frac{n \cdot S \cdot D_{\text{mm}}}{N \cdot f}$$

$$\text{DBH}_{\text{cm}} = \frac{W_{\text{mm}}}{10}$$

| Symbol | Variable name in CSV | Description |
|---|---|---|
| $n$ | `dbh_width` | Measured trunk width in pixels |
| $S$ | `sensor_width` | Camera sensor width in mm |
| $D_{\text{mm}}$ | `length` (×10) | Camera-to-subject distance in mm (`length` column is in cm) |
| $N$ | `image_width` | Full image width in pixels |
| $f$ | `focal_length` | Camera focal length in mm |

**Usage:** Edit the four path variables at the top of `utils/dbh_estimator.py` and run:

```python
JSON_FOLDER   = "../notebooks/seg_experiment/groundingdino_outputs"
IMAGE_FOLDER  = "../notebooks/data"
METADATA_CSV  = "dbh_csv.csv"    # columns: photo, length, sensor_width, focal_length
OUTPUT_CSV    = "estimated_dbh.csv"
```

```bash
python utils/dbh_estimator.py
```

Camera parameters (`sensor_width`, `focal_length`) can be found in your camera's EXIF data or specification sheet. `length` is the measured distance from the camera to the trunk in centimetres. If `actual_dbh` is included in the metadata CSV, the script also prints mean absolute error in cm and %.

---

## Utility Modules

| File | Description |
|---|---|
| `utils/dbh_estimator.py` | **All-in-one post-processing script**: extracts pixel DBH from JSON, reads image dimensions, merges with field metadata, applies pinhole formula, outputs `estimated_dbh.csv` |
| `utils/dbh_metric_converter.py` | Low-level pixel-to-cm conversion function (used internally by `dbh_estimator.py`) |
| `utils/PCA_Implementation.py` | Standalone PCA experiment on tilted tree images (GroundingDINO + SAM 2, largest trunk only) |
| `utils/NMS_Technique.py` | Standalone NMS experiment on branchy tree images (all detected trunks) |
| `utils/mask_dictionary_model.py` | `MaskDictionaryModel` / `ObjectInfo` dataclasses for multi-frame mask tracking with IoU-based ID assignment |
| `utils/supervision_utils.py` | Custom colour map for `supervision` annotation |
| `utils/track_utils.py` | Tracking utilities for video use cases |
| `utils/video_utils.py` | Video frame I/O helpers |
| `utils/common_utils.py` | Shared utility functions |
| `utils/groudingdino_florence_comparison_generator.py` | Side-by-side output comparison between GroundingDINO and Florence-2 detections |
| `utils/grounding_dino_florence_output_analysis.py` | Quantitative analysis of detection outputs |

---

## Installation

### Docker (Recommended)

Docker provides a fully pre-built environment — all C++ extensions (SAM 2, Grounding DINO) are compiled inside the image. No manual `conda`, CUDA toolkit, or compiler setup is needed on the host.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or Docker Engine (Linux)
- **GPU:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and `nvidia-smi` working on the host
- NVIDIA driver ≥ 525

> **CPU-only:** Omit `--gpus all` from every `docker run` command below and drop `--build-arg USE_CUDA=1` at build time. All three scripts fall back to CPU automatically.

**1. Clone the repository**

```bash
git clone https://github.com/Forest-Economy-Alliance/MonoDBH
cd MonoDBH
```

**2. Build the image**

```bash
# GPU build (CUDA 12.1) — set TORCH_ARCH to match your GPU:
#   7.5 = RTX 20xx  |  8.0 = A100  |  8.6 = RTX 30xx  |  8.9 = RTX 40xx
docker build -t monodbh --build-arg USE_CUDA=1 --build-arg TORCH_ARCH="8.6" .

# CPU-only build:
docker build -t monodbh .
```

**3. Download model checkpoints** *(one-time — files are saved to the host via volume mounts)*

```bash
# Linux / macOS
docker run --rm \
  -v "$(pwd)/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" \
  -v "$(pwd)/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" \
  monodbh bash -c "cd checkpoints && bash download_ckpts.sh && cd ../gdino_checkpoints && bash download_ckpts.sh"
```

```powershell
# Windows (PowerShell)
docker run --rm `
  -v "${PWD}/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" `
  -v "${PWD}/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" `
  monodbh bash -c "cd checkpoints && bash download_ckpts.sh && cd ../gdino_checkpoints && bash download_ckpts.sh"
```

**4. Place your images** in `notebooks/data/` (or `notebooks/tilted_trees/`, `notebooks/branches_data/`) on the host.

**5. Run a DBH script**

```bash
# Linux / macOS (GPU)
docker run --gpus all --rm \
  -v "$(pwd)/notebooks:/home/appuser/Grounded-SAM-2/notebooks" \
  -v "$(pwd)/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" \
  -v "$(pwd)/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" \
  monodbh python grounded_sam2_base.py
```

```powershell
# Windows (PowerShell + GPU)
docker run --gpus all --rm `
  -v "${PWD}/notebooks:/home/appuser/Grounded-SAM-2/notebooks" `
  -v "${PWD}/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" `
  -v "${PWD}/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" `
  monodbh python grounded_sam2_base.py
```

Replace `grounded_sam2_base.py` with `grounded_sam2_florence2.py` or `sam2_segmentation.py` as needed. Outputs are written to `notebooks/seg_experiment/` on your host.

**6. Run the metric conversion**

```bash
# Linux / macOS
docker run --rm \
  -v "$(pwd)/notebooks:/home/appuser/Grounded-SAM-2/notebooks" \
  -v "$(pwd)/utils:/home/appuser/Grounded-SAM-2/utils" \
  monodbh python utils/dbh_estimator.py
```

```powershell
# Windows (PowerShell)
docker run --rm `
  -v "${PWD}/notebooks:/home/appuser/Grounded-SAM-2/notebooks" `
  -v "${PWD}/utils:/home/appuser/Grounded-SAM-2/utils" `
  monodbh python utils/dbh_estimator.py
```

> The conda-based manual setup is documented below for users who prefer not to use Docker.

---

### Virtual Environment — DBH Estimator Only

If you only need to run `utils/dbh_estimator.py` (the pixel-to-cm conversion step) and not the segmentation scripts, a plain Python virtual environment is sufficient. No GPU, PyTorch, SAM 2, or Grounding DINO installation is needed.

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install pillow>=9.4.0 pandas>=2.2.0
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install pillow>=9.4.0 pandas>=2.2.0
```

Or install all non-PyTorch runtime dependencies at once:

```bash
pip install -r requirements.txt
```

Edit the four path variables at the top of `utils/dbh_estimator.py`, then run:

```bash
python utils/dbh_estimator.py
```

See [INSTALL.md](INSTALL.md#virtual-environment--dbh-estimator-only) for full details.

---

### Prerequisites *(manual / conda)*

| Requirement | Version |
|---|---|
| OS | Linux (recommended) / WSL2 on Windows |
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.3.1 |
| torchvision | ≥ 0.18.1 |
| CUDA toolkit | 12.1 (must match PyTorch build) — **optional**, CPU-only also works |
| GPU VRAM | ≥ 8 GB recommended (16 GB for Florence-2-large); not required for CPU |

> **CPU-only (no GPU):** All three scripts detect the device automatically via `torch.cuda.is_available()` and fall back to CPU if no GPU is present. Inference is functional but **significantly slower** — expect several minutes per image on CPU compared to seconds on GPU. SAM 2 automatic mask generation is the slowest on CPU; consider reducing `points_per_side` to `8` when running without a GPU.

> **Windows users:** Use [WSL2 with Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install). Native Windows is not supported for the CUDA extensions required by Grounding DINO.

---

### Step-by-step Setup

**1. Clone this repository**

```bash
git clone https://github.com/Forest-Economy-Alliance/MonoDBH
cd MonoDBH
```

**2. Create and activate a conda environment**

```bash
conda create -n monodbh python=3.10 -y
conda activate monodbh
```

**3. Install PyTorch with CUDA 12.1**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4. Set CUDA_HOME (required for Grounding DINO C++ extension)**

```bash
export CUDA_HOME=/usr/local/cuda-12.1
```

**5. Install SAM 2**

```bash
pip install -e ".[notebooks]"
```

To skip the optional CUDA post-processing extension:

```bash
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

**6. Install Grounding DINO**

```bash
pip install --no-build-isolation -e grounding_dino
```

**7. Install remaining dependencies**

```bash
pip install supervision transformers pandas opencv-python pillow torchvision
```

---

### Download Checkpoints

**SAM 2 checkpoints:**

```bash
cd checkpoints
bash download_ckpts.sh
cd ..
```

**Grounding DINO checkpoints:**

```bash
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
```

**Florence-2** is downloaded automatically from HuggingFace Hub on first run (`microsoft/Florence-2-large`). Ensure you have internet access or pre-cache it:

```bash
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)"
```

---

## Usage

### Data preparation

Place your input images in the appropriate folder:

```
notebooks/data/          ← general tree images
notebooks/tilted_trees/  ← leaning/tilted tree images
notebooks/branches_data/ ← images with heavy branching
```

Supported formats: `.jpg`, `.jpeg`, `.png`

---

### Running Grounding DINO + SAM 2

```bash
python grounded_sam2_base.py
```

Output is written to `notebooks/seg_experiment/groundingdino_outputs/`.

To change the input folder or detection prompt, edit these variables at the top of the script:

```python
input_folder = "notebooks/data/"
text = "tree trunk. hand"   # period-separated class names for Grounding DINO
```

---

### Running Florence-2 + SAM 2

```bash
python grounded_sam2_florence2.py
```

Output is written to `notebooks/seg_experiment/florence_outputs/`.

Florence-2 requires ~16 GB VRAM in float16 mode. If you encounter OOM errors, reduce image resolution or switch to the `Florence-2-base` variant by changing `FLORENCE2_ID` in the script.

---

### Running SAM 2 Automatic

```bash
python sam2_segmentation.py
```

Output is written to `notebooks/seg_experiment/sam2/`.

Reduce `points_per_side` further (e.g. `8`) if you hit VRAM limits:

```python
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=8,   # lower = less VRAM, less accurate
    ...
)
```

---

### Converting Pixel DBH to Centimetres

Once a segmentation script has produced JSON output files, run `dbh_estimator.py` to go from JSON → estimated DBH in one command:

```python
# Edit these four variables at the top of utils/dbh_estimator.py:
JSON_FOLDER   = "../notebooks/seg_experiment/groundingdino_outputs"
IMAGE_FOLDER  = "../notebooks/data"
METADATA_CSV  = "dbh_csv.csv"   # photo, length (cm), sensor_width (mm), focal_length (mm)
OUTPUT_CSV    = "estimated_dbh.csv"
```

```bash
python utils/dbh_estimator.py
```

Camera parameters (`sensor_width`, `focal_length`) can be found in your camera's EXIF data or spec sheet. `length` is the camera-to-trunk distance in centimetres. Add an `actual_dbh` column to the metadata CSV to get a MAE validation summary.

---

## Data Collection Procedure

### Field Protocol

For each tree in the study the following steps were performed in the field:

1. **GPS location** — recorded the geographic coordinates of the tree.
2. **Ground-truth DBH** — measured the trunk diameter at breast height (1.3 m above ground) with a diameter tape and recorded it as the reference value.
3. **Image capture** — photographed the tree trunk straight-on. One hand was placed flat against the trunk at the time of capture, with the known hand length serving as an in-image scale reference for depth estimation. **Using a hand is not required for your own images** — any known reference object works, or you can skip the reference entirely and supply the camera-to-trunk distance directly (see the note below).
4. **Camera-to-trunk distance** — recorded the distance from the camera lens to the trunk surface in centimetres. This value (`length`) is a required input to the pixel-to-cm conversion formula; it does **not** need to be derived from the image.

> **Note for custom data:** You do not need to place your hand in the image. What is essential is that you know the physical distance between the camera and the trunk at the moment of capture. This can be measured with a tape measure, laser rangefinder, or any other method. Record it alongside the image filename in your field CSV so the conversion notebook can use it.

### Dataset Statistics

| Property | Value |
|---|---|
| Total images | 978 |
| Devices used | 5 |
| Collection sites | Hyderabad & West Bengal, India |

---

## Data and Folder Structure

```
MonoDBH/
├── grounded_sam2_base.py          # Main pipeline: Grounding DINO + SAM 2
├── grounded_sam2_florence2.py     # Alternative pipeline: Florence-2 + SAM 2
├── sam2_segmentation.py           # Baseline: SAM 2 automatic (no detector)
│
├── utils/
│   ├── dbh_metric_converter.py    # Pixel → cm conversion
│   ├── PCA_Implementation.py      # Tilted tree experiment
│   ├── NMS_Technique.py           # Branchy tree experiment
│   ├── mask_dictionary_model.py   # Multi-frame mask tracking dataclasses
│   ├── supervision_utils.py       # Annotation colours
│   ├── track_utils.py             # Tracking helpers
│   ├── video_utils.py             # Video I/O
│   └── common_utils.py            # Shared utilities
│
├── checkpoints/                   # SAM 2 model weights (not tracked by git)
├── gdino_checkpoints/             # Grounding DINO weights (not tracked by git)
├── sam2/                          # SAM 2 source (from facebookresearch/sam2)
├── grounding_dino/                # Grounding DINO source (from IDEA-Research)
│
├── notebooks/
│   ├── data/                      # Input images (place your images here)
│   ├── tilted_trees/              # Tilted tree images
│   ├── branches_data/             # Branchy tree images
│   └── seg_experiment/            # All outputs
│       ├── groundingdino_outputs/
│       ├── florence_outputs/
│       └── sam2/
│
├── LICENSE
├── CODE_OF_CONDUCT.md
└── CONTRIBUTING.md
```

---

## Output Format

Each processed image produces:

**Annotated image** (`pca_<filename>.jpg`):
- Coloured instance masks per detection
- Bounding boxes with class labels
- Blue line: PCA principal axis of the trunk
- Red dot: centroid of the trunk mask
- Green line: DBH diameter (perpendicular to principal axis)

**JSON sidecar** (`<filename>.json`): array of detection records, e.g.:

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
      "left": [118, 467],
      "right": [332, 467]
    },
    "diameter_center_y": 467
  }
]
```

| Field | Description |
|---|---|
| `bbox` | Bounding box `[x1, y1, x2, y2]` in pixels |
| `area` | Mask area in pixels² |
| `trunk_angle_deg` | Angle of trunk axis from horizontal (°); 90° = perfectly vertical |
| `diameter_px` | DBH measurement in pixels |
| `diameter_line_coords` | Pixel coordinates of the two endpoints of the diameter line |
| `diameter_center_y` | Y-coordinate at which the diameter was measured |

---

## Reproducibility Notes

This code accompanies a journal paper.

> **Full end-to-end reproducibility documentation is in [docs/reproducibility_notes.md](docs/reproducibility_notes.md).**
>
> That document covers: the field data collection protocol, which script to use for each tree type (upright / tilted / branchy), how PCA corrects for trunk tilt, how NMS handles complex scenes, the complete JSON output schema, how `dbh_estimator.py` extracts pixel DBH widths and runs metric conversion in a single command, the structure of the field measurement CSV (`dbh_csv.csv`), the conversion formula with a worked numerical example, and a full troubleshooting section.

Key parameter summary for quick reference:

| Parameter | Value | Where set |
|---|---|---|
| Grounding DINO detection threshold | `0.25` | `grounded_sam2_base.py` |
| NMS IoU threshold | `0.2` | all detection-based scripts |
| SAM 2 checkpoint | `sam2.1_hiera_large` | all scripts |
| SAM 2 automatic `points_per_side` | `16` | `sam2_segmentation.py` |
| PCA row band for start point | ±5 px around bounding box midpoint | all detection-based scripts |
| Connected-component filter | largest component only per trunk mask | all detection-based scripts |
| Python / PyTorch / CUDA | 3.10 / 2.3.1 / 12.1 | — |

---

## Third-Party Credits and Licences

This project builds directly on the following open-source works. We are grateful to their authors for making the code publicly available.

### SAM 2 — Segment Anything Model 2
**Authors:** Meta AI Research (FAIR)  
**Repository:** https://github.com/facebookresearch/sam2  
**Paper:** *SAM 2: Segmentation in Images and Videos* — Ravi et al., 2024. [`arXiv:2408.00714`](https://arxiv.org/abs/2408.00714)  
**Licence:** Apache 2.0 — see [`LICENSE_sam2`](LICENSE_sam2)

SAM 2 source code is included in the `sam2/` directory of this repository under its original licence.

---

### Grounding DINO
**Authors:** IDEA-Research  
**Repository:** https://github.com/IDEA-Research/GroundingDINO  
**Paper:** *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection* — Liu et al., 2023. [`arXiv:2303.05499`](https://arxiv.org/abs/2303.05499)  
**Licence:** Apache 2.0 — see [`LICENSE_groundingdino`](LICENSE_groundingdino)

Grounding DINO source code is included in the `grounding_dino/` directory of this repository under its original licence.

---

### Grounded SAM 2 (pipeline concept and reference implementation)
**Authors:** IDEA-Research — Tianhe Ren, Shuo Shen et al.  
**Repository:** https://github.com/IDEA-Research/Grounded-SAM-2  
**Paper:** *Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks* — Ren et al., 2024. [`arXiv:2401.14159`](https://arxiv.org/abs/2401.14159)  
**Licence:** Apache 2.0

The concept of combining Grounding DINO with SAM 2 for prompted segmentation is taken from the Grounded SAM 2 framework by IDEA-Research. Our DBH pipeline adapts and extends this approach for tree trunk measurement.

---

### Florence-2
**Authors:** Microsoft  
**Repository:** https://huggingface.co/microsoft/Florence-2-large  
**Paper:** *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks* — Xiao et al., 2023. [`arXiv:2311.06242`](https://arxiv.org/abs/2311.06242)  
**Licence:** MIT

Used via HuggingFace Transformers as an alternative open-vocabulary detection back-end (`<OPEN_VOCABULARY_DETECTION>` task).

---

### supervision
**Authors:** Roboflow  
**Repository:** https://github.com/roboflow/supervision  
**Licence:** MIT

Used for annotation rendering (bounding boxes, masks, labels).

---

## Licence

This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.

Third-party components retain their original licences:

| Component | Licence | File |
|---|---|---|
| SAM 2 | Apache 2.0 | [LICENSE_sam2](LICENSE_sam2) |
| Grounding DINO | Apache 2.0 | [LICENSE_groundingdino](LICENSE_groundingdino) |
| CCTorch | Apache 2.0 | [LICENSE_cctorch](LICENSE_cctorch) |
| Florence-2 | MIT | (HuggingFace Hub) |
| supervision | MIT | (pip package) |

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this standard. Please report unacceptable behaviour to the project maintainers.

---

## Contributing

Contributions, bug reports, and suggestions are welcome. Please open a GitHub Issue or Pull Request. Before contributing, read [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Citation

If you use MonoDBH in your research, please cite the associated paper (BibTeX to be updated upon publication):

```bibtex
@article{monodbh2026,
  title     = {Monocular Tree DBH Estimation via Segmentation and Scale Reference},
  author    = {[Authors]},
  journal   = {[Journal Name]},
  year      = {2026},
}
```

Please also cite the underlying models:

```bibtex
@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and others},
  journal = {arXiv:2408.00714},
  year    = {2024}
}

@inproceedings{liu2023grounding,
  title     = {Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author    = {Liu, Shilong and others},
  booktitle = {ECCV},
  year      = {2024}
}

@article{ren2024grounded,
  title   = {Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks},
  author  = {Ren, Tianhe and others},
  journal = {arXiv:2401.14159},
  year    = {2024}
}

@article{xiao2023florence2,
  title   = {Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  author  = {Xiao, Bin and others},
  journal = {arXiv:2311.06242},
  year    = {2023}
}
```
