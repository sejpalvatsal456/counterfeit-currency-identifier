# Fake Indian Note Detection

A prototype web app and Python backend for classifying Indian banknote photos as `real` or `fake`.

The project has two parts:

- A browser UI to upload or capture a note photo.
- A FastAPI backend that crops the note, preprocesses the image, and predicts with a TensorFlow CNN.

## Project Structure

```text
dataset/
  real/      Put genuine note photos here.
  fake/      Put counterfeit/fake note photos here.
models/
  note_model.keras   Created after training.
src/
  features.py         Image loading and note cropping.
  train_model.py      TensorFlow training script.
  api.py              FastAPI prediction server.
index.html            Web UI.
app.js                Browser camera/upload logic.
styles.css            UI styles.
```

## Install

Use Python 3.10, 3.11, or 3.12 on Windows if possible.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

If you want OCR-based serial-number extraction, also install:

```powershell
python -m pip install easyocr
```

## Add Training Data

See `DATASET_GUIDE.md` for dataset sources and safe collection advice.

Add labelled photos like this:

```text
dataset/real/real_001.jpg
dataset/real/real_002.jpg
dataset/fake/fake_001.jpg
dataset/fake/fake_002.jpg
```

Keeping one denomination per model is still recommended, because different denominations have different color and layout patterns.

## Train

```powershell
python -m src.train_model --dataset dataset --output models/note_model.keras
```

This creates:

```text
models/note_model.keras
```

The script loads images from `dataset/real` and `dataset/fake`, applies data augmentation, trains a MobileNetV2-based binary classifier, and saves the best model.

## Run The Detection App

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open this on your computer:

```text
http://127.0.0.1:8000
```

For phone testing, connect your phone and computer to the same Wi-Fi network and open:

```text
http://YOUR_COMPUTER_IP:8000
```

## How It Works

- `src/features.py` loads images, detects the note region, corrects perspective, resizes to 224×224, and normalizes pixel values.
- `src/train_model.py` builds a TensorFlow dataset, freezes a MobileNetV2 feature extractor, adds a classifier head, and trains a binary model.
- `src/api.py` loads `models/note_model.keras` on startup and exposes `POST /predict` for uploaded note images.
- `app.js` sends the photo to `/predict` and displays real/fake probabilities plus detected serial number.

## Important Limits

This is a prototype helper, not a forensic detector. Accuracy depends heavily on the dataset quality. Real deployment should use expert-labelled counterfeit data, strict validation, and separate testing on cameras not used during training.
