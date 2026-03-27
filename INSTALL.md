# Installation & Quick-Start Guide — MonoDBH

This document covers everything you need to install the environment, download model weights, prepare your images, and run the DBH estimation scripts.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Docker Quick-Start (Recommended)](#docker-quick-start-recommended)
- [Virtual Environment — DBH Estimator Only](#virtual-environment--dbh-estimator-only)
- [1. Clone the Repository](#1-clone-the-repository) *(manual / conda)*
- [2. Create a Conda Environment](#2-create-a-conda-environment)
- [3. Install PyTorch with CUDA](#3-install-pytorch-with-cuda)
- [4. Set CUDA\_HOME](#4-set-cuda_home)
- [5. Install SAM 2](#5-install-sam-2)
- [6. Install Grounding DINO](#6-install-grounding-dino)
- [7. Install Remaining Dependencies](#7-install-remaining-dependencies)
- [8. Download Model Checkpoints](#8-download-model-checkpoints)
- [9. Prepare Your Images](#9-prepare-your-images)
- [10. Run the DBH Scripts](#10-run-the-dbh-scripts)
  - [Grounding DINO + SAM 2](#grounding-dino--sam-2-primary-pipeline)
  - [Florence-2 + SAM 2](#florence-2--sam-2-alternative-pipeline)
  - [SAM 2 Automatic](#sam-2-automatic-baseline-no-detector)
  - [Pixel-to-cm Conversion](#pixel-to-cm-conversion)
- [Common Installation Issues](#common-installation-issues)

---

## System Requirements

| Requirement | Recommended version |
|---|---|
| OS | Linux / WSL2 on Windows (Ubuntu 20.04 or 22.04) |
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.3.1 |
| torchvision | ≥ 0.18.1 |
| CUDA toolkit | 12.1 (must match your PyTorch build) — **optional**, CPU fallback is supported |
| GPU VRAM | ≥ 8 GB (16 GB recommended for Florence-2-large); not required for CPU |

> **CPU-only (no GPU):** All scripts fall back to CPU automatically. Inference is significantly slower — several minutes per image. Skip the `CUDA_HOME` step and the Grounding DINO C++ extension build when going CPU-only (use `SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"`).

> **Windows users:** Native Windows is not supported for the CUDA extensions required by Grounding DINO. Use [WSL2 with Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install).

---

## Docker Quick-Start (Recommended)

Docker is the easiest way to run MonoDBH. All C++ extensions (SAM 2, Grounding DINO CUDA ops) are pre-compiled inside the image, so there is nothing to configure on the host beyond Docker itself.

### Prerequisites

| Requirement | Notes |
|---|---|
| Docker Desktop (Windows/macOS) or Docker Engine (Linux) | [Install guide](https://docs.docker.com/get-docker/) |
| NVIDIA Container Toolkit | [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — required for GPU access |
| NVIDIA driver ≥ 525 | Verify with `nvidia-smi` |

> **CPU-only:** All three scripts detect the device automatically. Drop `--build-arg USE_CUDA=1` at build time and omit `--gpus all` from run commands.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Forest-Economy-Alliance/MonoDBH
cd MonoDBH
```

---

### Step 2 — Build the Docker image

The root `Dockerfile` installs PyTorch 2.3.1 + CUDA 12.1, SAM 2, and Grounding DINO in a single image.

**GPU build** (set `TORCH_ARCH` for your GPU):

| GPU family | `TORCH_ARCH` value |
|---|---|
| RTX 20xx (Turing) | `7.5` |
| A100 (Ampere) | `8.0` |
| RTX 30xx (Ampere) | `8.6` |
| RTX 40xx (Ada) | `8.9` |

```bash
docker build -t monodbh \
  --build-arg USE_CUDA=1 \
  --build-arg TORCH_ARCH="8.6" \
  .
```

**CPU-only build:**

```bash
docker build -t monodbh .
```

> The first build compiles the Grounding DINO CUDA extension and may take 10–20 minutes. Subsequent builds use Docker’s layer cache.

---

### Step 3 — Download model checkpoints *(one-time)*

Run the download scripts inside a temporary container. The files are written to the **host** via volume mounts and will be reused on every future `docker run`.

**Linux / macOS:**
```bash
docker run --rm \
  -v "$(pwd)/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" \
  -v "$(pwd)/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" \
  monodbh bash -c \
    "cd checkpoints && bash download_ckpts.sh && \
     cd ../gdino_checkpoints && bash download_ckpts.sh"
```

**Windows (PowerShell):**
```powershell
docker run --rm `
  -v "${PWD}/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" `
  -v "${PWD}/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" `
  monodbh bash -c `
    "cd checkpoints && bash download_ckpts.sh && cd ../gdino_checkpoints && bash download_ckpts.sh"
```

Expected downloads: `sam2.1_hiera_large.pt` (~850 MB) and the Grounding DINO weights.

---

### Step 4 — Prepare your images

Create input directories on the host and place your images inside:

```bash
mkdir -p notebooks/data
mkdir -p notebooks/tilted_trees
mkdir -p notebooks/branches_data
```

| Folder | Use for |
|---|---|
| `notebooks/data/` | General upright tree images |
| `notebooks/tilted_trees/` | Leaning or tilted trees |
| `notebooks/branches_data/` | Trees with heavy branching |

---

### Step 5 — Run a DBH script

Mount the `notebooks/` and checkpoint directories so data flows between host and container.

**Linux / macOS (GPU):**
```bash
docker run --gpus all --rm \
  -v "$(pwd)/notebooks:/home/appuser/Grounded-SAM-2/notebooks" \
  -v "$(pwd)/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" \
  -v "$(pwd)/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" \
  monodbh python grounded_sam2_base.py
```

**Windows (PowerShell + GPU):**
```powershell
docker run --gpus all --rm `
  -v "${PWD}/notebooks:/home/appuser/Grounded-SAM-2/notebooks" `
  -v "${PWD}/checkpoints:/home/appuser/Grounded-SAM-2/checkpoints" `
  -v "${PWD}/gdino_checkpoints:/home/appuser/Grounded-SAM-2/gdino_checkpoints" `
  monodbh python grounded_sam2_base.py
```

| Script | Command |
|---|---|
| Grounding DINO + SAM 2 | `python grounded_sam2_base.py` |
| Florence-2 + SAM 2 | `python grounded_sam2_florence2.py` |
| SAM 2 Automatic | `python sam2_segmentation.py` |

Output files (annotated images + JSON sidecars) are written to `notebooks/seg_experiment/` on your host machine.

---

### Step 6 — Run the metric conversion

**Linux / macOS:**
```bash
docker run --rm \
  -v "$(pwd)/notebooks:/home/appuser/Grounded-SAM-2/notebooks" \
  -v "$(pwd)/utils:/home/appuser/Grounded-SAM-2/utils" \
  monodbh python utils/dbh_estimator.py
```

**Windows (PowerShell):**
```powershell
docker run --rm `
  -v "${PWD}/notebooks:/home/appuser/Grounded-SAM-2/notebooks" `
  -v "${PWD}/utils:/home/appuser/Grounded-SAM-2/utils" `
  monodbh python utils/dbh_estimator.py
```

> Edit the four path variables at the top of `utils/dbh_estimator.py` before running (see [Step 10 — Run the DBH Scripts](#10-run-the-dbh-scripts) for details).

---

### Common Docker Issues

<details>
<summary><strong>docker: Error response from daemon: could not select device driver "nvidia"</strong></summary>
<br/>

The NVIDIA Container Toolkit is not installed or the Docker daemon has not been restarted after installation. Follow the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and run `sudo systemctl restart docker`.
</details>

<details>
<summary><strong>Windows: bind mount path not found / permission denied</strong></summary>
<br/>

Ensure Docker Desktop has access to the drive. Go to **Settings → Resources → File Sharing** and add the drive letter (e.g. `C:`). Then retry the `docker run` command.
</details>

<details>
<summary><strong>Build fails with "gcc: error: unrecognized command-line option '-arch'"</strong></summary>
<br/>

The `TORCH_ARCH` value contains an architecture not supported by the installed GCC. Use only the compute capability for your GPU (e.g. `"8.6"` for RTX 30xx) instead of a semicolon-separated list.
</details>

---

## Virtual Environment — DBH Estimator Only

If you only need to run `utils/dbh_estimator.py` (the pixel-to-cm conversion step) and **not** the segmentation scripts, you can skip torch, SAM 2, and Grounding DINO entirely. A plain Python virtual environment with two lightweight packages is all that is needed.

> The segmentation scripts (`grounded_sam2_base.py`, `grounded_sam2_florence2.py`, `sam2_segmentation.py`) require PyTorch, the SAM 2 package, and Grounding DINO. Use Docker or the full conda setup for those.

**1. Create and activate the virtual environment**

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

**2. Install the estimator dependencies**

```bash
pip install pillow>=9.4.0 pandas>=2.2.0
```

Or install all non-PyTorch runtime dependencies at once using the `requirements.txt`:

```bash
pip install -r requirements.txt
```

**3. Edit the config paths and run**

Open `utils/dbh_estimator.py` and update the four path variables at the top of the file:

```python
JSON_FOLDER   = "../notebooks/seg_experiment/groundingdino_outputs"
IMAGE_FOLDER  = "../notebooks/data"
METADATA_CSV  = "../notebooks/metadata.csv"
OUTPUT_CSV    = "../notebooks/estimated_dbh.csv"
```

Then run:

```bash
python utils/dbh_estimator.py
```

**4. Deactivate the environment when done**

```bash
deactivate
```

---

## 1. Clone the Repository *(manual / conda)*

```bash
git clone https://github.com/<your-org>/MonoDBH.git
cd MonoDBH
```

---

## 2. Create a Conda Environment

```bash
conda create -n monodbh python=3.10 -y
conda activate monodbh
```

---

## 3. Install PyTorch with CUDA

Install PyTorch 2.3.1 with CUDA 12.1 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.3.1+cu121  True
```

---

## 4. Set CUDA\_HOME

The Grounding DINO C++ / CUDA extension requires `CUDA_HOME` to be set:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
```

Verify:

```bash
python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)"
# Expected: True  /usr/local/cuda-12.1
```

> If `CUDA_HOME` is not found automatically, check `nvcc --version` and adjust the path above to match.

---

## 5. Install SAM 2

Install SAM 2 from the root of this repository:

```bash
pip install -e ".[notebooks]"
```

To skip the optional CUDA post-processing extension (safe to skip — only affects tiny mask hole-filling):

```bash
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

---

## 6. Install Grounding DINO

```bash
pip install --no-build-isolation -e grounding_dino
```

> The `--no-build-isolation` flag is required because Grounding DINO's CUDA extension needs the already-installed PyTorch headers at build time.

---

## 7. Install Remaining Dependencies

```bash
pip install supervision transformers pandas opencv-python pillow
```

Florence-2 will be downloaded automatically on first run. To pre-cache it now:

```bash
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
"
```

---

## 8. Download Model Checkpoints

**SAM 2.1 checkpoints** (required by all three scripts):

```bash
cd checkpoints
bash download_ckpts.sh
cd ..
```

This downloads `sam2.1_hiera_large.pt` (~850 MB) which is the checkpoint used by all scripts.

**Grounding DINO checkpoints** (required by `grounded_sam2_base.py` and the utils scripts):

```bash
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
```

Verify the files exist:

```bash
ls checkpoints/sam2.1_hiera_large.pt
ls gdino_checkpoints/
```

> Florence-2 weights are downloaded automatically from HuggingFace Hub — no manual step needed.

---

## 9. Prepare Your Images

Create the input directories and place your tree images inside:

```bash
mkdir -p notebooks/data
mkdir -p notebooks/tilted_trees
mkdir -p notebooks/branches_data
```

| Folder | Use for |
|---|---|
| `notebooks/data/` | General upright tree images (primary dataset) |
| `notebooks/tilted_trees/` | Leaning or tilted trees |
| `notebooks/branches_data/` | Trees with heavy branching / complex scenes |

**Image requirements:**
- Format: `.jpg`, `.jpeg`, or `.png`
- The scene should contain a tree trunk with a **human hand** held against it at breast height — the hand is the physical scale reference used for metric conversion.
- Recommended resolution: 1080p or higher for accurate pixel measurements.

---

## 10. Run the DBH Scripts

All scripts are run from the **root of the repository** with the conda environment active.

---

### Grounding DINO + SAM 2 (primary pipeline)

```bash
python grounded_sam2_base.py
```

- Reads from: `notebooks/data/`
- Writes to: `notebooks/seg_experiment/groundingdino_outputs/`
- Produces per-image annotated `.jpg` + `.json` sidecar with `diameter_px`, `trunk_angle_deg`, and diameter line coordinates.

To change the input folder or detection prompt, edit the top of the script:

```python
input_folder = "notebooks/data/"        # ← change to your folder
text = "tree trunk. hand"               # ← Grounding DINO class prompt (period-separated)
```

---

### Florence-2 + SAM 2 (alternative pipeline)

```bash
python grounded_sam2_florence2.py
```

- Reads from: `notebooks/data/`
- Writes to: `notebooks/seg_experiment/florence_outputs/`
- Requires ~16 GB VRAM in float16 mode. If you hit OOM errors, switch to `Florence-2-base`:

```python
FLORENCE2_ID = "microsoft/Florence-2-base"   # line ~12 of the script
```

---

### SAM 2 Automatic (baseline, no detector)

```bash
python sam2_segmentation.py
```

- Reads from: `notebooks/data/`
- Writes to: `notebooks/seg_experiment/sam2/`
- No detection model — assumes the **largest segmented region** is the tree trunk.
- Outputs a green mask overlay. No diameter line is drawn in this baseline.

If you hit VRAM limits, lower `points_per_side` in the script:

```python
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=8,   # default is 16; reduce to save VRAM
    ...
)
```

---

### Pixel-to-cm Conversion

After running any of the segmentation scripts, use `utils/dbh_metric_converter.py` to convert the pixel diameter measurements to real-world centimetres.

Prepare a CSV file with one row per image containing these columns:

| Column | Description | Units |
|---|---|---|
| `dbh_width` | Measured trunk width in pixels (from the JSON output) | px |
| `sensor_width` | Camera sensor width | mm |
| `length` | Camera-to-trunk distance | cm |
| `image_width` | Full image width in pixels | px |
| `focal_length` | Camera focal length | mm |

> Camera EXIF data (readable with `exiftool`) provides `sensor_width`, `image_width`, and `focal_length`. `length` must be measured in the field.

Run the conversion:

```python
from utils.dbh_metric_converter import estimate_dbh

df = estimate_dbh(
    input_csv="measurements.csv",
    output_csv="results_with_dbh.csv"
)
print(df[["dbh_width", "estimated_dbh"]])
```

The output CSV will have an `estimated_dbh` column in centimetres.

---

## Common Installation Issues

<details>
<summary>I got <code>ImportError: cannot import name '_C' from 'sam2'</code></summary>
<br/>

You haven't run the `pip install -e ".[notebooks]"` step, or it failed silently. Re-run it and check for errors. On some systems:

```bash
python setup.py build_ext --inplace
```
</details>

<details>
<summary>I got <code>MissingConfigException: Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'</code></summary>
<br/>

SAM 2 is not in your Python path. Re-run `pip install -e .` from the repo root. If it still fails:

```bash
export PYTHONPATH="/path/to/MonoDBH:${PYTHONPATH}"
```
</details>

<details>
<summary>I got <code>RuntimeError: Error(s) in loading state_dict for SAM2Base</code> with SAM 2.1 checkpoints</summary>
<br/>

You have an older SAM 2 installation. Reinstall cleanly:

```bash
pip uninstall -y SAM-2
pip install -e ".[notebooks]"
```
</details>

<details>
<summary>My installation failed with <code>CUDA_HOME environment variable is not set</code></summary>
<br/>

Set `CUDA_HOME` explicitly (see [Step 4](#4-set-cuda_home)) and retry. Also verify with:

```bash
python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
```

If it still fails, add `--no-build-isolation`:

```bash
pip install --no-build-isolation -e .
```
</details>

<details>
<summary>I got <code>undefined symbol: _ZN3c1015SmallVectorBaseIjE8grow_podEPKvmm</code></summary>
<br/>

Multiple conflicting PyTorch/CUDA versions in your environment. Use a fresh conda environment and install only one version of PyTorch (≥ 2.3.1) via pip.
</details>

<details>
<summary>I got <code>CUDA error: no kernel image is available for execution on the device</code></summary>
<br/>

The CUDA extension was compiled for a different GPU architecture. Set the target architecture explicitly before reinstalling:

```bash
export TORCH_CUDA_ARCH_LIST="9.0 8.0 8.6 8.9 7.0 7.5 6.0"
pip install -e ".[notebooks]"
```
</details>

<details>
<summary>I got <code>RuntimeError: No available kernel. Aborting execution.</code></summary>
<br/>

Flash Attention is not available on your GPU. In `sam2/modeling/sam/transformer.py`, replace:

```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
```

with:

```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True
```
</details>

<details>
<summary>I got <code>Error compiling objects for extension</code> (Windows / unsupported MSVC)</summary>
<br/>

Your CUDA and Visual Studio versions are incompatible. Add `-allow-unsupported-compiler` to the `nvcc` flags in `setup.py`:

```python
"nvcc": [
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-allow-unsupported-compiler"
],
```

Alternatively, using WSL2 avoids this issue entirely.
</details>

<details>
<summary>Florence-2 runs out of memory (OOM)</summary>
<br/>

Florence-2-large requires ~16 GB VRAM in float16. Options:

1. Use `Florence-2-base` instead (change `FLORENCE2_ID` in `grounded_sam2_florence2.py`).
2. Resize your input images to a lower resolution before processing.
3. Add `torch.cuda.empty_cache()` between images if processing a large batch.
</details>

<details>
<summary>No detections found for my images</summary>
<br/>

- For Grounding DINO: lower the detection threshold from `0.25` to `0.15` in `grounded_sam2_base.py`.
- Ensure the text prompt matches what is visible: `"tree trunk. hand"` — the period separates classes in Grounding DINO syntax.
- Check that the morphological close preprocessing step isn't destroying fine features (reduce kernel from `5×5` to `3×3` for high-resolution images).
- For SAM 2 automatic: the largest segment may not be the trunk if background objects are large — this is a known limitation of that baseline.
</details>

Then, install SAM 2 from the root of this repository via
```bash
pip install -e ".[notebooks]"
```

Note that you may skip building the SAM 2 CUDA extension during installation via environment variable `SAM2_BUILD_CUDA=0`, as follows:
```bash
# skip the SAM 2 CUDA extension
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```
This would also skip the post-processing step at runtime (removing small holes and sprinkles in the output masks, which requires the CUDA extension), but shouldn't affect the results in most cases.

### Building the SAM 2 CUDA extension

By default, we allow the installation to proceed even if the SAM 2 CUDA extension fails to build. (In this case, the build errors are hidden unless using `-v` for verbose output in `pip install`.)

If you see a message like `Skipping the post-processing step due to the error above` at runtime or `Failed to build the SAM 2 CUDA extension due to the error above` during installation, it indicates that the SAM 2 CUDA extension failed to build in your environment. In this case, **you can still use SAM 2 for both image and video applications**. The post-processing step (removing small holes and sprinkles in the output masks) will be skipped, but this shouldn't affect the results in most cases.

If you would like to enable this post-processing step, you can reinstall SAM 2 on a GPU machine with environment variable `SAM2_BUILD_ALLOW_ERRORS=0` to force building the CUDA extension (and raise errors if it fails to build), as follows
```bash
pip uninstall -y SAM-2 && \
rm -f ./sam2/*.so && \
SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e ".[notebooks]"
```

Note that PyTorch needs to be installed first before building the SAM 2 CUDA extension. It's also necessary to install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation. (This should typically be CUDA 12.1 if you follow the default installation command.) After installing the CUDA toolkits, you can check its version via `nvcc --version`.

Please check the section below on common installation issues if the CUDA extension fails to build during installation or load at runtime.

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
I got `ImportError: cannot import name '_C' from 'sam2'`
</summary>
<br/>

This is usually because you haven't run the `pip install -e ".[notebooks]"` step above or the installation failed. Please install SAM 2 first, and see the other issues if your installation fails.

In some systems, you may need to run `python setup.py build_ext --inplace` in the SAM 2 repo root as suggested in https://github.com/facebookresearch/sam2/issues/77.
</details>

<details>
<summary>
I got `MissingConfigException: Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'`
</summary>
<br/>

This is usually because you haven't run the `pip install -e .` step above, so `sam2` isn't in your Python's `sys.path`. Please run this installation step. In case it still fails after the installation step, you may try manually adding the root of this repo to `PYTHONPATH` via
```bash
export SAM2_REPO_ROOT=/path/to/sam2  # path to this repo
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"
```
to manually add `sam2_configs` into your Python's `sys.path`.

</details>

<details>
<summary>
I got `RuntimeError: Error(s) in loading state_dict for SAM2Base` when loading the new SAM 2.1 checkpoints
</summary>
<br/>

This is likely because you have installed a previous version of this repo, which doesn't have the new modules to support the SAM 2.1 checkpoints yet. Please try the following steps:

1. pull the latest code from the `main` branch of this repo
2. run `pip uninstall -y SAM-2` to uninstall any previous installations
3. then install the latest repo again using `pip install -e ".[notebooks]"`

In case the steps above still don't resolve the error, please try running in your Python environment the following
```python
from sam2.modeling import sam2_base

print(sam2_base.__file__)
```
and check whether the content in the printed local path of `sam2/modeling/sam2_base.py` matches the latest one in https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py (e.g. whether your local file has `no_obj_embed_spatial`) to indentify if you're still using a previous installation.

</details>

<details>
<summary>
My installation failed with `CUDA_HOME environment variable is not set`
</summary>
<br/>

This usually happens because the installation step cannot find the CUDA toolkits (that contain the NVCC compiler) to build a custom CUDA kernel in SAM 2. Please install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) or the version that matches the CUDA version for your PyTorch installation. If the error persists after installing CUDA toolkits, you may explicitly specify `CUDA_HOME` via
```
export CUDA_HOME=/usr/local/cuda  # change to your CUDA toolkit path
```
and rerun the installation.

Also, you should make sure
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
print `(True, a directory with cuda)` to verify that the CUDA toolkits are correctly set up.

If you are still having problems after verifying that the CUDA toolkit is installed and the `CUDA_HOME` environment variable is set properly, you may have to add the `--no-build-isolation` flag to the pip command:
```
pip install --no-build-isolation -e .
```

</details>

<details>
<summary>
I got `undefined symbol: _ZN3c1015SmallVectorBaseIjE8grow_podEPKvmm` (or similar errors)
</summary>
<br/>

This usually happens because you have multiple versions of dependencies (PyTorch or CUDA) in your environment. During installation, the SAM 2 library is compiled against one version library while at run time it links against another version. This might be due to that you have different versions of PyTorch or CUDA installed separately via `pip` or `conda`. You may delete one of the duplicates to only keep a single PyTorch and CUDA version.

In particular, if you have a lower PyTorch version than 2.3.1, it's recommended to upgrade to PyTorch 2.3.1 or higher first. Otherwise, the installation script will try to upgrade to the latest PyTorch using `pip`, which could sometimes lead to duplicated PyTorch installation if you have previously installed another PyTorch version using `conda`.

We have been building SAM 2 against PyTorch 2.3.1 internally. However, a few user comments (e.g. https://github.com/facebookresearch/sam2/issues/22, https://github.com/facebookresearch/sam2/issues/14) suggested that downgrading to PyTorch 2.1.0 might resolve this problem. In case the error persists, you may try changing the restriction from `torch>=2.3.1` to `torch>=2.1.0` in both [`pyproject.toml`](pyproject.toml) and [`setup.py`](setup.py) to allow PyTorch 2.1.0.
</details>

<details>
<summary>
I got `CUDA error: no kernel image is available for execution on the device`
</summary>
<br/>

A possible cause could be that the CUDA kernel is somehow not compiled towards your GPU's CUDA [capability](https://developer.nvidia.com/cuda-gpus). This could happen if the installation is done in an environment different from the runtime (e.g. in a slurm system).

You can try pulling the latest code from the SAM 2 repo and running the following
```
export TORCH_CUDA_ARCH_LIST=9.0 8.0 8.6 8.9 7.0 7.2 7.5 6.0`
```
to manually specify the CUDA capability in the compilation target that matches your GPU.
</details>

<details>
<summary>
I got `RuntimeError: No available kernel. Aborting execution.` (or similar errors)
</summary>
<br/>

This is probably because your machine doesn't have a GPU or a compatible PyTorch version for Flash Attention (see also https://discuss.pytorch.org/t/using-f-scaled-dot-product-attention-gives-the-error-runtimeerror-no-available-kernel-aborting-execution/180900 for a discussion in PyTorch forum). You may be able to resolve this error by replacing the line
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
```
in [`sam2/modeling/sam/transformer.py`](sam2/modeling/sam/transformer.py) with
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True
```
to relax the attention kernel setting and use other kernels than Flash Attention.
</details>

<details>
<summary>
I got `Error compiling objects for extension`
</summary>
<br/>

You may see error log of:
> unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

This is probably because your versions of CUDA and Visual Studio are incompatible. (see also https://stackoverflow.com/questions/78515942/cuda-compatibility-with-visual-studio-2022-version-17-10 for a discussion in stackoverflow).<br> 
You may be able to fix this by adding the `-allow-unsupported-compiler` argument to `nvcc` after L48 in the [setup.py](https://github.com/facebookresearch/sam2/blob/main/setup.py). <br>
After adding the argument, `get_extension()` will look like this:
```python
def get_extensions():
    srcs = ["sam2/csrc/connected_components.cu"]
    compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-allow-unsupported-compiler"  # Add this argument
        ],
    }
    ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    return ext_modules
```
</details>
