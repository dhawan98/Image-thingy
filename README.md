# Eye-Tracking Heatmap Pipeline

This repository contains the code and annotation layout needed to reproduce the round-level heatmaps from the Unity screenshots, eye-tracking CSVs, and camera matrix file.

Raw data and generated outputs are intentionally not committed.

## What it generates

- One corrected heatmap overlay per experimental phase in `outputs/heatmaps/rounds/`.
- One segmentation/debug preview per phase in `outputs/debug/`.
- Classification CSVs that show how many gaze hits were assigned to each main object.

## Repo layout

- `.gitignore`
- `data/raw/camera/camera_matrices.txt`
- `data/raw/images/round_*.png`
- `data/raw/eye_tracking/*.csv`
- `data/annotations/object_layout.json`
- `scripts/generate_heatmaps.py`
- `outputs/heatmaps/rounds/`
- `outputs/debug/`
- `outputs/metrics/`

## What is committed

- Heatmap generation code
- Object layout / segmentation configuration
- Minimal dependency list
- Reproduction instructions

## What is not committed

- Raw eye-tracking CSVs
- Source screenshots
- Camera file
- Generated outputs
- Zip archives

## Recreate outputs

1. Create the expected folders:

```bash
mkdir -p data/raw/camera data/raw/images data/raw/eye_tracking outputs
```

2. Place the raw assets into the repo using these filenames:

- `data/raw/camera/camera_matrices.txt`
- `data/raw/images/round_01_NONE.png`
- `data/raw/images/round_02_P_L.png`
- `data/raw/images/round_03_P_M.png`
- `data/raw/images/round_04_P_H.png`
- `data/raw/images/round_05_O_L.png`
- `data/raw/images/round_06_O_M.png`
- `data/raw/images/round_07_O_H.png`
- `data/raw/images/round_08_PO_L.png`
- `data/raw/images/round_09_PO_M.png`
- `data/raw/images/round_10_PO_H.png`
- all eye-tracking CSVs inside `data/raw/eye_tracking/`

3. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

4. Run the generator:

```bash
python3 scripts/generate_heatmaps.py
```

5. Review outputs:

- `outputs/heatmaps/rounds/`
- `outputs/debug/`
- `outputs/metrics/`

## Notes

- Aggregation is keyed by `Phase` (`NONE`, `P_L`, `P_M`, etc.), not the numeric `Round`, because participants experienced the 10 conditions in different round orders.
- The script looks for `Exact Object`, `ExactObject`, and then `GazeHitObject`; the current files use `GazeHitObject`.
- Real-environment `Quad` hits are projected into image space only for object classification, then accumulated on the segmented masks for `quad`, `real_table`, `real_robot`, and `real_brick`.
- Virtual-environment objects are assigned directly from the exact object name and accumulated on the segmented masks for `virtual_table`, `virtual_robot`, and `virtual_brick`.
