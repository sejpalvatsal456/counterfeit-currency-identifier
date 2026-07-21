# How The ML Model Works

This project detects whether an Indian currency note photo is likely `real` or `fake` using a TensorFlow image classifier.

It learns patterns from labelled images placed in:

```text
dataset/real/
dataset/fake/
```

## 1. Training Data

Collect photos of notes and label them by folder:

- `dataset/real/` contains genuine note photos.
- `dataset/fake/` contains fake, counterfeit, or approved fake-like note photos.

For best results, keep one denomination per training set, for example INR 500.

## 2. Image Preprocessing

The file `src/features.py` loads each note image, finds the largest note-like contour, performs perspective correction, and resizes the result to 224×224 pixels.

The image is then converted from BGR to RGB and normalized to float values in `[0, 1]`.

This is the input the CNN uses.

## 3. Model Architecture

The file `src/train_model.py` builds a TensorFlow model using:

- `MobileNetV2` as a frozen feature extractor,
- `GlobalAveragePooling2D`,
- `Dropout`,
- a dense ReLU layer,
- a sigmoid output for binary real/fake prediction.

Data augmentation includes random flip, rotation, zoom, and contrast changes.

## 4. Training The Classifier

`src/train_model.py`:

1. Loads images from `dataset/real` and `dataset/fake`.
2. Converts them into TensorFlow tensors.
3. Shuffles and splits the dataset 80/20 into training and validation sets.
4. Trains the model for up to 30 epochs.
5. Saves the best model to `models/note_model.keras`.

This is a CNN-based approach, not a hand-crafted feature classifier.

## 5. Prediction Flow

When the frontend sends a photo to `POST /predict`:

1. `src/api.py` reads the uploaded image bytes.
2. `src.features.preprocess_image` crops and normalizes the note image.
3. `src/api.py` loads `models/note_model.keras` at startup.
4. The model predicts a probability value.
5. The API returns:

```json
{
  "prediction": "real" | "fake",
  "confidence": 0.92,
  "real_probability": 0.92,
  "fake_probability": 0.08,
  "serial_number": "AB1234567" | null
}
```

The web UI displays the model verdict, the confidence percentage, and any detected serial number.

## 6. Browser-side Quality Checks

The UI in `index.html` / `app.js` also computes browser-side metrics such as:

- photo brightness,
- sharpness,
- expected note aspect ratio,
- denomination color hint.

These are visual guidance checks only. The backend model currently predicts from the raw cropped image alone.

## 7. Why Dataset Quality Matters

Training on high-quality, well-labelled images is essential.

If all fake images share one background and all real images share another, the model may learn the background instead of note appearance.

Good data should include:

- varying lighting,
- different cameras,
- different backgrounds,
- different note conditions,
- held-out validation images.

## 8. Current Limitations

This prototype cannot prove authenticity. It does not verify physical security features such as:

- watermark paper under transmitted light,
- raised intaglio print feel,
- optically variable ink,
- UV or magnetic properties.

It is a visual classifier that learns from labeled phone-camera images.

## 9. Best Practical Use

Use this project as a prototype and learning tool:

```text
photo -> crop and preprocess -> CNN classifier -> real/fake probability
```

Do not use it as final proof that a note is genuine or counterfeit.
