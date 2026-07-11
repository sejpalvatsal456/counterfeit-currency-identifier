from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


DENOMINATION_PROFILES = {
    "10": {"aspect": 123 / 63, "color": np.array([116, 75, 54], dtype=np.float32)},
    "20": {"aspect": 129 / 63, "color": np.array([185, 139, 68], dtype=np.float32)},
    "50": {"aspect": 135 / 66, "color": np.array([75, 132, 125], dtype=np.float32)},
    "100": {"aspect": 142 / 66, "color": np.array([128, 94, 151], dtype=np.float32)},
    "200": {"aspect": 146 / 66, "color": np.array([198, 151, 62], dtype=np.float32)},
    "500": {"aspect": 150 / 66, "color": np.array([112, 118, 92], dtype=np.float32)},
    "2000": {"aspect": 166 / 66, "color": np.array([167, 91, 143], dtype=np.float32)},
}

FEATURE_NAMES = [
    "aspect_ratio",
    "aspect_error",
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "thread_score",
    "watermark_score",
    "see_through_score",
    "micro_text_score",
    "dominant_color_distance",
    "h_mean",
    "s_mean",
    "v_mean",
    "h_std",
    "s_std",
    "v_std",
]


@dataclass(frozen=True)
class FeatureResult:
    vector: np.ndarray
    diagnostics: dict[str, float]


def load_image_from_bytes(payload: bytes) -> np.ndarray:
    image_array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image. Use a clear JPG or PNG photo.")
    return image


def extract_features(image: np.ndarray, denomination: str) -> FeatureResult:
    if denomination not in DENOMINATION_PROFILES:
        raise ValueError(f"Unsupported denomination: {denomination}")

    note = _crop_largest_note_like_region(image)
    note = cv2.resize(note, (640, 280), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(note, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 60, 160)
    profile = DENOMINATION_PROFILES[denomination]

    aspect_ratio = note.shape[1] / note.shape[0]
    aspect_error = abs(aspect_ratio - profile["aspect"]) / profile["aspect"]
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 128.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)
    edge_density = float(np.mean(edges > 0))

    thread_score = _security_thread_score(gray)
    watermark_score = _watermark_score(gray)
    see_through_score = _see_through_register_score(edges)
    micro_text_score = _micro_text_score(gray)

    mean_bgr = np.mean(note.reshape(-1, 3), axis=0)[::-1].astype(np.float32)
    color_distance = float(np.linalg.norm(mean_bgr - profile["color"]) / 255.0)
    hsv_mean = np.mean(hsv.reshape(-1, 3), axis=0) / np.array([179.0, 255.0, 255.0])
    hsv_std = np.std(hsv.reshape(-1, 3), axis=0) / np.array([90.0, 128.0, 128.0])

    values = np.array(
        [
            aspect_ratio,
            aspect_error,
            brightness,
            contrast,
            sharpness,
            edge_density,
            thread_score,
            watermark_score,
            see_through_score,
            micro_text_score,
            color_distance,
            *hsv_mean.tolist(),
            *hsv_std.tolist(),
        ],
        dtype=np.float32,
    )

    diagnostics = {name: float(value) for name, value in zip(FEATURE_NAMES, values)}
    return FeatureResult(vector=values, diagnostics=diagnostics)


def _crop_largest_note_like_region(image: np.ndarray) -> np.ndarray:
    max_width = 1200
    scale = min(1.0, max_width / image.shape[1])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < resized.shape[0] * resized.shape[1] * 0.05:
            continue
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
        aspect = max(width, height) / min(width, height)
        if 1.5 <= aspect <= 3.2:
            candidates.append((area, rect))

    if not candidates:
        return resized

    _, best_rect = max(candidates, key=lambda item: item[0])
    box = cv2.boxPoints(best_rect).astype(np.float32)
    return _four_point_transform(resized, box)


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = _order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    width = max(1, int(max(width_a, width_b)))
    height = max(1, int(max(height_a, height_b)))

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _security_thread_score(gray: np.ndarray) -> float:
    height, width = gray.shape
    middle = gray[:, int(width * 0.18) : int(width * 0.82)]
    vertical_edges = cv2.Sobel(middle, cv2.CV_64F, 1, 0, ksize=3)
    column_strength = np.mean(np.abs(vertical_edges), axis=0)
    return float(np.clip(np.max(column_strength) / 80.0, 0.0, 1.0))


def _watermark_score(gray: np.ndarray) -> float:
    height, width = gray.shape
    left_region = gray[int(height * 0.12) : int(height * 0.82), int(width * 0.03) : int(width * 0.28)]
    if left_region.size == 0:
        return 0.0
    local_contrast = cv2.Laplacian(left_region, cv2.CV_64F).var()
    tonal_range = float(np.percentile(left_region, 92) - np.percentile(left_region, 8))
    return float(np.clip((tonal_range / 90.0) + (local_contrast / 900.0), 0.0, 1.0))


def _see_through_register_score(edges: np.ndarray) -> float:
    height, width = edges.shape
    region = edges[int(height * 0.18) : int(height * 0.5), int(width * 0.25) : int(width * 0.5)]
    if region.size == 0:
        return 0.0
    density = float(np.mean(region > 0))
    return float(np.clip(density / 0.18, 0.0, 1.0))


def _micro_text_score(gray: np.ndarray) -> float:
    height, width = gray.shape
    lower_band = gray[int(height * 0.58) : int(height * 0.88), int(width * 0.18) : int(width * 0.88)]
    if lower_band.size == 0:
        return 0.0
    high_freq = cv2.Laplacian(lower_band, cv2.CV_64F).var()
    return float(np.clip(high_freq / 1400.0, 0.0, 1.0))
