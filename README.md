# Fake Indian Note Detection

A trainable prototype for detecting whether an Indian currency note photo is real or fake.

The project now has two parts:

- A phone-friendly web UI for clicking or uploading a note photo.
- A Python computer-vision backend that extracts note features, trains a classifier, and returns real/fake probabilities.

## Project Structure

```text
dataset/
  real/      Put genuine note photos here.
  fake/      Put counterfeit/fake note photos here.
models/
  note_model.joblib   Created after training.
src/
  features.py         OpenCV feature extraction.
  train_model.py      Training script.
  api.py              FastAPI prediction server.
index.html            Web UI.
app.js                Browser camera/upload logic.
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

See `DATASET_GUIDE.md` for dataset sources and safe collection advice.

Add labelled photos like this:

```text
dataset/real/real_001.jpg
dataset/real/real_002.jpg
dataset/fake/fake_001.jpg
dataset/fake/fake_002.jpg
```

Use the same denomination for one model. For example, train one model for INR 500 notes first. Use many photos from different phones, distances, angles, and lighting conditions. A useful first target is at least 100 real and 100 fake photos.

## Train

```powershell
python -m src.train_model --denomination 500
```

This creates:

```text
models/note_model.joblib
```

The script prints a classification report, confusion matrix, and test file predictions.

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

## Features Extracted

The model uses OpenCV to crop the likely note area and extract:

- Note aspect-ratio match for the selected denomination.
- Brightness, contrast, and sharpness.
- Edge density.
- Security-thread score from vertical edge structure.
- Watermark-region score from tonal variation.
- See-through-register region score.
- Micro-text/high-frequency print score.
- Dominant denomination color distance.
- HSV color statistics.

These features are passed into a scikit-learn Random Forest classifier.

## Important Limits

This is a trainable prototype, not a certified forensic detector. Accuracy depends heavily on your labelled dataset. Real deployment should use expert-labelled images, strict validation, separate denomination models, and testing on phones/cameras that were not used during training.
