#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
EYE_DIR = DATA_DIR / "eye_tracking"
IMAGE_DIR = DATA_DIR / "images"
CAMERA_PATH = DATA_DIR / "camera" / "camera_matrices.txt"
LAYOUT_PATH = ROOT / "data" / "annotations" / "object_layout.json"

OUTPUT_DIR = ROOT / "outputs"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps" / "rounds"
DEBUG_DIR = OUTPUT_DIR / "debug"
METRICS_DIR = OUTPUT_DIR / "metrics"

PHASE_ORDER = [
    "NONE",
    "P_L",
    "P_M",
    "P_H",
    "O_L",
    "O_M",
    "O_H",
    "PO_L",
    "PO_M",
    "PO_H",
]

PHASE_TO_IMAGE = {
    "NONE": "round_01_NONE.png",
    "P_L": "round_02_P_L.png",
    "P_M": "round_03_P_M.png",
    "P_H": "round_04_P_H.png",
    "O_L": "round_05_O_L.png",
    "O_M": "round_06_O_M.png",
    "O_H": "round_07_O_H.png",
    "PO_L": "round_08_PO_L.png",
    "PO_M": "round_09_PO_M.png",
    "PO_H": "round_10_PO_H.png",
}

PHASE_TO_LABEL = {
    phase: f"round_{index:02d}_{phase}" for index, phase in enumerate(PHASE_ORDER, start=1)
}

OBJECT_COLUMN_CANDIDATES = ["Exact Object", "ExactObject", "GazeHitObject"]

VIRTUAL_OBJECT_GROUPS = {
    "virtual_brick": {"DT_Brick_1"},
    "virtual_table": {"RayTabletopSurface"},
    "virtual_robot": {
        "base_link_0",
        "shoulder_link_0",
        "arm_link_0",
        "forearm_link_0",
        "elbow_link_0",
        "wrist_link_0",
        "hand_link_0",
        "G1_MainSupport_0",
        "G1_ClampLeft_0",
        "G1_ClampRight_0",
        "G1_Rod_0",
        "G1_ServoHead_0",
    },
}

MAIN_OBJECT_ORDER = [
    "quad",
    "real_table",
    "real_robot",
    "real_brick",
    "virtual_table",
    "virtual_robot",
    "virtual_brick",
]

OBJECT_TITLES = {
    "quad": "Quad",
    "real_table": "Real Table",
    "real_robot": "Real Robot",
    "real_brick": "Real Brick",
    "virtual_table": "Virtual Table",
    "virtual_robot": "Virtual Robot",
    "virtual_brick": "DT Brick",
}


@dataclass(frozen=True)
class CameraConfig:
    view_projection: np.ndarray
    width: int
    height: int


def ensure_output_dirs() -> None:
    for path in [HEATMAP_DIR, DEBUG_DIR, METRICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_camera_config(camera_path: Path) -> CameraConfig:
    text = camera_path.read_text()
    size_match = re.search(r"Screenshot Size:\s*(\d+)x(\d+)", text)
    matrix_match = re.search(
        r"=== Combined View-Projection Matrix \(4x4\) ===\s*((?:\s*\[[^\n]+\]\n){4})",
        text,
    )
    if not size_match or not matrix_match:
        raise ValueError(f"Could not parse camera config from {camera_path}")

    width, height = map(int, size_match.groups())
    rows = []
    for line in matrix_match.group(1).strip().splitlines():
        values = [float(value.strip()) for value in line.strip()[1:-1].split(",")]
        rows.append(values)

    return CameraConfig(view_projection=np.array(rows, dtype=np.float64), width=width, height=height)


def resolve_object_column() -> str:
    sample_csv = next(EYE_DIR.glob("*.csv"))
    columns = pd.read_csv(sample_csv, nrows=0).columns.tolist()
    for column in OBJECT_COLUMN_CANDIDATES:
        if column in columns:
            return column
    raise ValueError(f"Could not find an object column in {sample_csv}. Available columns: {columns}")


def project_world_to_image(points_xyz: np.ndarray, camera: CameraConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homogeneous = np.hstack([points_xyz, np.ones((len(points_xyz), 1), dtype=np.float64)])
    clip = homogeneous @ camera.view_projection.T
    w = clip[:, 3]
    valid = np.abs(w) > 1e-6

    ndc = np.zeros((len(points_xyz), 2), dtype=np.float64)
    ndc[valid, 0] = clip[valid, 0] / w[valid]
    ndc[valid, 1] = clip[valid, 1] / w[valid]

    pixel_x = (ndc[:, 0] + 1.0) * 0.5 * camera.width
    pixel_y = (1.0 - ndc[:, 1]) * 0.5 * camera.height

    in_frame = (
        valid
        & np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & (pixel_x >= 0)
        & (pixel_x < camera.width)
        & (pixel_y >= 0)
        & (pixel_y < camera.height)
    )
    return pixel_x, pixel_y, in_frame


def load_layout() -> dict:
    return json.loads(LAYOUT_PATH.read_text())


def rect_mask(shape: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def ellipse_mask(shape: tuple[int, int], center: tuple[int, int], axes: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def shape_to_mask(shape: tuple[int, int], spec: dict) -> np.ndarray:
    if spec["type"] == "rect":
        return rect_mask(shape, spec["x1"], spec["y1"], spec["x2"], spec["y2"])
    if spec["type"] == "ellipse":
        return ellipse_mask(shape, tuple(spec["center"]), tuple(spec["axes"]))
    raise ValueError(f"Unsupported shape type: {spec['type']}")


def detect_brick_mask(
    image_bgr: np.ndarray,
    roi: dict,
    default_box: dict,
    min_area: int,
    max_area: int,
    pad: tuple[int, int],
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    roi_img = image_bgr[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    purple_mask = cv2.inRange(
        hsv,
        np.array([120, 35, 70], dtype=np.uint8),
        np.array([170, 255, 255], dtype=np.uint8),
    )

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(purple_mask)
    best_box: tuple[int, int, int, int] | None = None
    best_area = -1
    for index in range(1, component_count):
        left, top, box_w, box_h, area = map(int, stats[index])
        if min_area <= area <= max_area and area > best_area:
            best_area = area
            best_box = (x1 + left, y1 + top, x1 + left + box_w, y1 + top + box_h)

    if best_box is None:
        best_box = (default_box["x1"], default_box["y1"], default_box["x2"], default_box["y2"])

    pad_x, pad_y = pad
    bx1, by1, bx2, by2 = best_box
    return rect_mask(
        (height, width),
        max(0, bx1 - pad_x),
        max(0, by1 - pad_y),
        min(width - 1, bx2 + pad_x),
        min(height - 1, by2 + pad_y),
    )


def build_object_masks(image_bgr: np.ndarray, layout: dict) -> dict[str, np.ndarray]:
    shape = image_bgr.shape[:2]
    masks = {
        name: shape_to_mask(shape, spec)
        for name, spec in layout["shared_regions"].items()
    }

    masks["real_brick"] = detect_brick_mask(
        image_bgr=image_bgr,
        roi=layout["brick_detection"]["real_brick_roi"],
        default_box=layout["brick_detection"]["real_brick_default"],
        min_area=10,
        max_area=120,
        pad=(12, 10),
    )
    masks["virtual_brick"] = detect_brick_mask(
        image_bgr=image_bgr,
        roi=layout["brick_detection"]["virtual_brick_roi"],
        default_box=layout["brick_detection"]["virtual_brick_default"],
        min_area=150,
        max_area=1400,
        pad=(12, 10),
    )

    # Make the regions disjoint so each classified hit contributes to only one visible object area.
    masks["real_table"] = cv2.subtract(masks["real_table"], masks["real_robot"])
    masks["real_table"] = cv2.subtract(masks["real_table"], masks["real_brick"])
    masks["quad"] = cv2.subtract(masks["quad"], masks["real_table"])
    masks["quad"] = cv2.subtract(masks["quad"], masks["real_robot"])
    masks["quad"] = cv2.subtract(masks["quad"], masks["real_brick"])

    masks["virtual_table"] = cv2.subtract(masks["virtual_table"], masks["virtual_robot"])
    masks["virtual_table"] = cv2.subtract(masks["virtual_table"], masks["virtual_brick"])
    return masks


def mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.mean()), int(ys.mean())


def add_header(image_bgr: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    header_height = 96
    canvas = np.full((height + header_height, width, 3), 255, dtype=np.uint8)
    canvas[header_height:] = image_bgr
    cv2.putText(canvas, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (70, 70, 70), 2, cv2.LINE_AA)
    return canvas


def draw_region_outlines(image_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    annotated = image_bgr.copy()
    for object_name in MAIN_OBJECT_ORDER:
        mask = masks[object_name]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (40, 210, 60), 2)
        cx, cy = mask_centroid(mask)
        cv2.putText(
            annotated,
            OBJECT_TITLES[object_name],
            (max(10, cx - 72), max(24, cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
    return annotated


def read_projected_hits(camera: CameraConfig, object_column: str) -> pd.DataFrame:
    projected_frames: list[pd.DataFrame] = []
    usecols = ["ParticipantID", "Phase", object_column, "GazeHitX", "GazeHitY", "GazeHitZ"]

    for csv_path in sorted(EYE_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        df["ParticipantID"] = df["ParticipantID"].astype(str)
        df["Phase"] = df["Phase"].astype(str)
        df = df[df["Phase"].isin(PHASE_ORDER)].copy()
        if df.empty:
            continue

        valid_hit = df[object_column].notna() & ~(
            (df["GazeHitX"] == 0) & (df["GazeHitY"] == 0) & (df["GazeHitZ"] == 0)
        )
        hits = df.loc[valid_hit, ["ParticipantID", "Phase", object_column, "GazeHitX", "GazeHitY", "GazeHitZ"]].copy()
        if hits.empty:
            continue

        xyz = hits[["GazeHitX", "GazeHitY", "GazeHitZ"]].to_numpy(dtype=np.float64)
        x_px, y_px, in_frame = project_world_to_image(xyz, camera)
        hits = hits.loc[in_frame, ["ParticipantID", "Phase", object_column]].copy()
        hits.rename(columns={object_column: "exact_object"}, inplace=True)
        hits["exact_object"] = hits["exact_object"].astype(str)
        hits["x_px"] = x_px[in_frame]
        hits["y_px"] = y_px[in_frame]
        projected_frames.append(hits)

    if not projected_frames:
        raise RuntimeError("No valid projected gaze hits were produced from the raw eye-tracking data.")

    return pd.concat(projected_frames, ignore_index=True)


def classify_real_quad_hit(x_px: float, y_px: float, masks: dict[str, np.ndarray]) -> str | None:
    x_idx = int(np.clip(round(x_px), 0, masks["quad"].shape[1] - 1))
    y_idx = int(np.clip(round(y_px), 0, masks["quad"].shape[0] - 1))

    if masks["real_brick"][y_idx, x_idx] > 0:
        return "real_brick"
    if masks["real_robot"][y_idx, x_idx] > 0:
        return "real_robot"
    if masks["real_table"][y_idx, x_idx] > 0:
        return "real_table"
    if masks["quad"][y_idx, x_idx] > 0:
        return "quad"
    return None


def classify_main_object(exact_object: str, x_px: float, y_px: float, masks: dict[str, np.ndarray]) -> str | None:
    for object_name, aliases in VIRTUAL_OBJECT_GROUPS.items():
        if exact_object in aliases:
            return object_name

    if exact_object == "Quad":
        return classify_real_quad_hit(x_px, y_px, masks)

    return None


def classify_hits_by_phase(
    projected_hits: pd.DataFrame,
    phase_masks: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    classified_rows = []
    for phase in PHASE_ORDER:
        phase_hits = projected_hits[projected_hits["Phase"] == phase]
        masks = phase_masks[phase]
        for row in phase_hits.itertuples(index=False):
            main_object = classify_main_object(row.exact_object, row.x_px, row.y_px, masks)
            if main_object is None:
                continue
            classified_rows.append(
                {
                    "ParticipantID": row.ParticipantID,
                    "Phase": row.Phase,
                    "exact_object": row.exact_object,
                    "main_object": main_object,
                    "x_px": row.x_px,
                    "y_px": row.y_px,
                }
            )

    if not classified_rows:
        raise RuntimeError("No gaze hits were classified into the main objects.")

    return pd.DataFrame(classified_rows)


def render_round_heatmap(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    object_counts: dict[str, int],
) -> np.ndarray:
    combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    density = np.zeros(image_bgr.shape[:2], dtype=np.float32)

    for object_name in MAIN_OBJECT_ORDER:
        mask = masks[object_name]
        combined_mask = np.maximum(combined_mask, mask)
        count = int(object_counts.get(object_name, 0))
        if count <= 0:
            continue
        area = max(1, int(np.count_nonzero(mask)))
        density[mask > 0] += count / area

    positive = density[density > 0]
    if positive.size == 0:
        return image_bgr.copy()

    heat = density / positive.max()
    heat = np.clip(heat, 0.0, 1.0)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)

    alpha = np.zeros_like(heat, dtype=np.float32)
    active = heat > 0
    alpha[active] = 0.28 + 0.52 * heat[active]

    overlay = image_bgr.astype(np.float32)
    overlay = overlay * (1.0 - alpha[..., None]) + color.astype(np.float32) * alpha[..., None]
    overlay = overlay.astype(np.uint8)

    for object_name in MAIN_OBJECT_ORDER:
        contours, _ = cv2.findContours(masks[object_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (35, 215, 55), 2)

    return overlay


def generate_outputs() -> None:
    ensure_output_dirs()

    layout = load_layout()
    camera = parse_camera_config(CAMERA_PATH)
    object_column = resolve_object_column()
    projected_hits = read_projected_hits(camera, object_column)

    phase_masks: dict[str, dict[str, np.ndarray]] = {}
    phase_images: dict[str, np.ndarray] = {}

    for phase in PHASE_ORDER:
        image_path = IMAGE_DIR / PHASE_TO_IMAGE[phase]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        phase_images[phase] = image_bgr
        phase_masks[phase] = build_object_masks(image_bgr, layout)

        debug_preview = draw_region_outlines(image_bgr, phase_masks[phase])
        debug_preview = add_header(debug_preview, f"{PHASE_TO_LABEL[phase]} | Segmentation Regions", "Green outlines are the object masks used for heatmap accumulation.")
        cv2.imwrite(str(DEBUG_DIR / f"{PHASE_TO_LABEL[phase]}_regions.png"), debug_preview)

    classified_hits = classify_hits_by_phase(projected_hits, phase_masks)

    metrics_rows = []
    for phase in PHASE_ORDER:
        image_bgr = phase_images[phase]
        masks = phase_masks[phase]
        phase_hits = classified_hits[classified_hits["Phase"] == phase]
        object_counts = phase_hits["main_object"].value_counts().to_dict()
        total_hits = int(len(phase_hits))

        for object_name in MAIN_OBJECT_ORDER:
            metrics_rows.append(
                {
                    "phase": phase,
                    "phase_label": PHASE_TO_LABEL[phase],
                    "main_object": object_name,
                    "object_title": OBJECT_TITLES[object_name],
                    "assigned_hits": int(object_counts.get(object_name, 0)),
                }
            )

        heatmap = render_round_heatmap(image_bgr, masks, object_counts)
        count_summary = ", ".join(
            f"{OBJECT_TITLES[name]}={int(object_counts.get(name, 0))}"
            for name in MAIN_OBJECT_ORDER
            if int(object_counts.get(name, 0)) > 0
        )
        if not count_summary:
            count_summary = "No main-object gaze hits classified."

        heatmap = add_header(
            heatmap,
            f"{PHASE_TO_LABEL[phase]} | Main-Object Heatmap",
            f"object_column={object_column} | classified_hits={total_hits} | {count_summary}",
        )
        cv2.imwrite(str(HEATMAP_DIR / f"{PHASE_TO_LABEL[phase]}_heatmap.png"), heatmap)

    classified_hits.to_csv(METRICS_DIR / "classified_hits.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(METRICS_DIR / "phase_object_counts.csv", index=False)

    print(f"Heatmaps generated successfully using object column: {object_column}")


if __name__ == "__main__":
    generate_outputs()
