from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.features import (
    load_image_from_bytes,
    preprocess_image,
)

from src.ocr import extract_serial_number


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "note_model.keras"


app = FastAPI(
    title="Fake Indian Note Detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (ROOT / "static").exists():
    app.mount(
        "/static",
        StaticFiles(directory=ROOT / "static"),
        name="static",
    )


cnn_model = None


@app.on_event("startup")
def load_model():

    global cnn_model

    if MODEL_PATH.exists():

        cnn_model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print("CNN model loaded.")

    else:

        print("Model not found.")


@app.get("/")
def index():

    return FileResponse(
        ROOT / "index.html"
    )


@app.get("/styles.css")
def styles():

    return FileResponse(
        ROOT / "styles.css"
    )


@app.get("/app.js")
def javascript():

    return FileResponse(
        ROOT / "app.js"
    )


@app.get("/health")
def health():

    return {
        "ok": True,
        "model_loaded": cnn_model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if cnn_model is None:

        raise HTTPException(
            status_code=503,
            detail="CNN model not found."
        )

    payload = await file.read()

    try:

        image = load_image_from_bytes(
            payload
        )

        cnn_input, cropped_note = preprocess_image(
            image,
            return_note=True,
        )

        serial_number = extract_serial_number(
            cropped_note
        )

        processed = preprocess_image(
            image
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    prediction = cnn_model.predict(
        np.expand_dims(processed, axis=0),
        verbose=0,
    )[0][0]

    fake_probability = float(
        1 - prediction
    )

    real_probability = float(
        prediction
    )

    label = (
        "real"
        if prediction >= 0.5
        else "fake"
    )

    confidence = max(
        real_probability,
        fake_probability,
    )

    return {

        "prediction": label,

        "confidence": round(
            confidence,
            4,
        ),

        "real_probability": round(
            real_probability,
            4,
        ),

        "fake_probability": round(
            fake_probability,
            4,
        ),

        "serial_number": serial_number,
    }