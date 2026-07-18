import cv2
import easyocr
import numpy as np
import re

reader = easyocr.Reader(
    ['en'],
    gpu=False        # Change to True if CUDA becomes available
)


def extract_serial_number(image: np.ndarray) -> str | None:
    """
    Extracts the serial number from an Indian banknote.

    Parameters
    ----------
    image : np.ndarray
        OpenCV BGR image

    Returns
    -------
    str | None
        Serial number if found, else None
    """

    img = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    results = reader.readtext(
        img,
        paragraph=False,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    detections = []

    for bbox, text, conf in results:

        pts = np.array(bbox, dtype=np.int32)

        x = pts[:,0].min()
        y = pts[:,1].min()
        w = pts[:,0].max() - x
        h = pts[:,1].max() - y

        detections.append({
            "text": text.strip(),
            "conf": conf,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        })

    serials = []

    for num in detections:

        if not re.fullmatch(r"\d{6,7}", num["text"]):
            continue

        best = None
        best_gap = float("inf")

        num_center = num["y"] + num["h"] / 2

        for left in detections:

            if left == num:
                continue

            if not re.search(r"[A-Z]", left["text"]):
                continue

            left_center = left["y"] + left["h"] / 2

            if abs(left_center - num_center) > max(left["h"], num["h"]) * 0.5:
                continue

            if left["x"] >= num["x"]:
                continue

            gap = num["x"] - (left["x"] + left["w"])

            if gap < 0 or gap > 250:
                continue

            if gap < best_gap:
                best_gap = gap
                best = left

        if best:

            serials.append({
                "serial": best["text"] + num["text"],
                "score": best["conf"] + num["conf"]
            })

    if not serials:
        return None

    serials.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return serials[0]["serial"]