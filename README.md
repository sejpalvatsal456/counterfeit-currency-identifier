# Fake Indian Note Detection

A trainable prototype for detecting whether an Indian currency note is real or fake, using photos of **both the front and back** of the note.

The project has two parts:

- A phone-friendly web UI for capturing or uploading a front photo and a back photo of a note.
- A Python computer-vision backend that extracts features from both sides, trains a classifier, and returns real/fake probabilities.

## Project Structure

```text
dataset/
  real/
    front/    Genuine note photos, front side.
    back/     Genuine note photos, back side (same filename as the matching front photo).
  fake/
    front/    Counterfeit/fake note photos, front side.
    back/     Counterfeit/fake note photos, back side (same filename as the matching front photo).
models/
  note_model.joblib   Created after training.
src/
  features.py         OpenCV feature extraction (front + back combined).
  train_model.py      Training script (pairs front/back images per note).
  api.py              FastAPI prediction server (accepts front_file + back_file).
  ocr.py              Serial number extraction from the front photo.
index.html            Web UI.
app.js                Browser camera/upload logic for front and back photos.
styles.css            UI styles.
```

## Install

Use Python 3.10, 3.11, or 3.12 on Windows if possible. These versions have reliable prebuilt wheels for NumPy, OpenCV, and scikit-learn.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip cache purge
python -m pip install --only-binary=:all: -r requirements.txt
```

If `python` points to Python 3.13 or newer and installation still fails, install Python 3.12 from python.org, recreate the virtual environment, and run the commands again.

## Add Training Data

**No dataset has been collected yet.** See `DATASET_GUIDE.md` for dataset sources and safe collection advice.

Every note needs a matched front and back photo saved under the same filename in the `front/` and `back/` subfolders:

```text
dataset/real/front/real_001.jpg
dataset/real/back/real_001.jpg
dataset/real/front/real_002.jpg
dataset/real/back/real_002.jpg
dataset/fake/front/fake_001.jpg
dataset/fake/back/fake_001.jpg
dataset/fake/front/fake_002.jpg
dataset/fake/back/fake_002.jpg
```

A front photo without a matching back photo (or vice versa) is skipped during training with a warning — it is not used.

Use the same denomination for one model. For example, train one model for INR 500 notes first. Use many photos from different phones, distances, angles, and lighting conditions. A useful first target is at least 100 real notes and 100 fake notes, each with front and back photos.

## Train

```powershell
python -m src.train_model --denomination 500
```

This creates:

```text
models/note_model.joblib
```

The script prints a classification report, confusion matrix, and test file predictions (showing the matched front/back pair for each test note).

## Run The Detection App

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open this on your computer:

```text
http://127.0.0.1:8000
```

For phone testing, connect phone and computer to the same Wi-Fi and open:

```text
http://YOUR_COMPUTER_IP:8000
```

In the app, capture or upload both the front and back of the note. The prediction request only fires once both photos are provided.

## Features Extracted

For each side (front and back) separately, the model uses OpenCV to crop the likely note area and extract:

- Note aspect-ratio match for the selected denomination.
- Brightness, contrast, and sharpness.
- Edge density.
- Security-thread score from vertical edge structure.
- Watermark-region score from tonal variation.
- See-through-register region score.
- Micro-text/high-frequency print score.
- Dominant denomination color distance.
- HSV color statistics.

The front-side and back-side feature vectors are concatenated into one combined vector per note and passed into a scikit-learn Random Forest classifier. Serial number OCR still runs on the front photo only.

## Important Limits

This is a trainable prototype, not a certified forensic detector. Accuracy depends heavily on your labelled dataset. Real deployment should use expert-labelled front/back image pairs, strict validation, separate denomination models, and testing on phones/cameras that were not used during training.