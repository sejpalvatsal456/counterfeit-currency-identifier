from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.ocr import extract_serial_number

from src.features import extract_features, load_image_from_bytes


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "note_model.joblib"

app = FastAPI(title="Fake Indian Note Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (ROOT / "static").exists():
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(ROOT / "styles.css")


@app.get("/app.js")
def javascript() -> FileResponse:
    return FileResponse(ROOT / "app.js")


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "model_trained": MODEL_PATH.exists()}


@app.get("/predict")
def predict_help() -> dict[str, str]:
    return {
        "message": "Use POST /predict with multipart form-data fields: file and denomination.",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    denomination: str = Form("500"),
) -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Add images to dataset/real and dataset/fake, then run python -m src.train_model.",
        )

    payload = await file.read()
    try:
        image = load_image_from_bytes(payload)
        serial_number = extract_serial_number(image)
        feature_result = extract_features(image, denomination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    trained_denomination = artifact.get("denomination")
    if trained_denomination != denomination:
        raise HTTPException(
            status_code=400,
            detail=f"Model was trained for INR {trained_denomination}, but request used INR {denomination}.",
        )

    vector = feature_result.vector.reshape(1, -1)
    probabilities = model.predict_proba(vector)[0]
    classes = list(model.classes_)
    real_index = classes.index(1)
    fake_index = classes.index(0)
    real_probability = float(probabilities[real_index])
    fake_probability = float(probabilities[fake_index])
    prediction = "real" if real_probability >= 0.5 else "fake"
    confidence = float(np.max(probabilities))

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "real_probability": round(real_probability, 4),
        "fake_probability": round(fake_probability, 4),
        "serial_number": serial_number,
        "denomination": denomination,
        "diagnostics": feature_result.diagnostics,
    }
