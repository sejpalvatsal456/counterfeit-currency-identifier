from __future__ import annotations

import cv2
import numpy as np

# MobileNetV2 input size
INPUT_SIZE = 224


def load_image_from_bytes(payload: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes into an OpenCV BGR image.
    """
    image_array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image.")

    return image


def preprocess_image(image, return_note=False):
    note = crop_note(image)

    cnn_input = cv2.cvtColor(note, cv2.COLOR_BGR2RGB)
    cnn_input = cnn_input.astype(np.float32) / 255.0

    if return_note:
        return cnn_input, note

    return cnn_input


def crop_note(image: np.ndarray) -> np.ndarray:
    """
    Detect the largest note-like contour and
    perform perspective correction.
    """

    max_width = 1200

    scale = min(1.0, max_width / image.shape[1])

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_area = 0

    image_area = resized.shape[0] * resized.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.05:
            continue

        rect = cv2.minAreaRect(contour)

        w, h = rect[1]

        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)

        if not (1.5 <= aspect <= 3.2):
            continue

        if area > best_area:
            best_area = area
            best = rect

    if best is None:

        return cv2.resize(
            resized,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    box = cv2.boxPoints(best)

    box = np.array(box, dtype=np.float32)

    return four_point_transform(
        resized,
        box,
    )


def order_points(points: np.ndarray) -> np.ndarray:

    rect = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)

    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def four_point_transform(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:

    rect = order_points(points)

    destination = np.array(
        [
            [0, 0],
            [INPUT_SIZE - 1, 0],
            [INPUT_SIZE - 1, INPUT_SIZE - 1],
            [0, INPUT_SIZE - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination,
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (INPUT_SIZE, INPUT_SIZE),
    )

    return warped


def load_training_image(path: str) -> np.ndarray:
    """
    Used while training.
    """

    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return preprocess_image(image)