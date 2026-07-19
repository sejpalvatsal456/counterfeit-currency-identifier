# How The ML Model Works

This project detects whether an Indian currency note is likely `real` or `fake` using a supervised machine-learning model that looks at **both the front and back photo of the note together**.

It does not magically know fake notes by itself. It learns patterns from labelled examples placed in:

```text
dataset/real/front/
dataset/real/back/
dataset/fake/front/
dataset/fake/back/
```

**No dataset exists yet.** These folders need to be created and filled with matched front/back photo pairs before training will work — see `DATASET_GUIDE.md`.

## 1. Training Data

You first collect photos of notes and label them:

- `dataset/real/front/` and `dataset/real/back/` contain genuine note photos.
- `dataset/fake/front/` and `dataset/fake/back/` contain fake, counterfeit, or approved fake-like note photos.

Each note's front and back photo must share the same filename (e.g. `front/500_001.jpg` and `back/500_001.jpg`) so the training script can pair them. A photo with no matching pair on the other side is skipped.

For best results, train one denomination at a time, for example only INR 500 notes.

The model compares the visual patterns in real-note images against fake-note images, using both sides of each note. If the dataset is small, the model may appear accurate but fail on new photos.

## 2. Feature Extraction

The file `src/features.py` extracts numeric features from every note photo.

When an image is loaded, the code tries to find the largest note-like rectangular region. It uses OpenCV edge detection and contours to crop the note area. Then it resizes the note image to a fixed size so all training images are measured consistently.

This runs **independently on the front photo and the back photo** of each note. Each side produces the same set of measurements:

```text
aspect_ratio
aspect_error
brightness
contrast
sharpness
edge_density
thread_score
watermark_score
see_through_score
micro_text_score
dominant_color_distance
h_mean
s_mean
v_mean
h_std
s_std
v_std
```

The front side's measurements and the back side's measurements are then concatenated (`front_aspect_ratio`, `back_aspect_ratio`, and so on) into a single combined feature vector for the note. This means the classifier can pick up on inconsistencies between the two sides, not just how convincing one side looks alone.

These are not final decisions. They are measurements that describe the note image.

## 3. Important Features

The features below are calculated the same way for the front photo and the back photo — each side gets its own score.

### Aspect Ratio

Indian notes have expected width and height ratios. The model checks whether the photographed note shape is close to the selected denomination.

Example:

```text
INR 500 expected ratio = 150 / 66
```

If the photo has a very different shape, it may indicate poor cropping, wrong denomination, or a fake-like sample. This is checked separately for the front and back crop.

### Brightness And Contrast

The model measures whether the note is too dark, too bright, or has enough visible print contrast.

Bad lighting can reduce accuracy, so the model needs training photos from different lighting conditions, for both sides of the note.

### Sharpness

Sharpness is calculated using the Laplacian variance method. A blurry image has fewer high-frequency details.

This matters because security features such as micro text, borders, and fine print are harder to detect in blurry images — on either side.

### Edge Density

Currency notes contain many printed lines, borders, symbols, numbers, and texture patterns. Edge density measures how much visual structure exists in the image.

Fake-like notes may have different print detail or lower-quality edges, on the front, the back, or both.

### Security Thread Score

The code looks for strong vertical structures in the central note region. This is a rough computer-vision estimate for a visible security thread.

It does not truly verify the physical thread. It only measures whether the image contains a thread-like vertical pattern.

### Watermark Score

The code checks the left-side watermark region for tonal variation and local contrast.

A real watermark is a physical feature seen by light passing through the note. A normal phone photo cannot always verify it correctly, so this score is only a visual hint.

### See-Through Register Score

The code looks at a region where see-through register patterns may appear and measures edge density. Because the see-through register pattern only fully lines up when front and back are aligned, having both sides' scores available lets the classifier learn from mismatches, not just each side individually.

This is also a rough visual feature, not a certified security test.

### Micro Text Score

The model measures high-frequency details in lower printed regions. Micro lettering and fine print create small sharp patterns.

If the note image is blurry, this score may be low even for a real note — front or back.

### Dominant Color Distance

Each denomination has an expected color profile. The model compares the average photo color with the expected color for the selected denomination, separately for the front and back photo.

Color alone is not enough because camera lighting can change it, but it helps as one feature.

## 4. Training The Classifier

The file `src/train_model.py` loads all image pairs from the dataset folders.

For each note:

1. Find the front image and its matching back image (same filename in `front/` and `back/`).
2. Skip the note if either side is missing.
3. Read both images using OpenCV.
4. Extract feature values from each side and concatenate them using `extract_dual_features`.
5. Assign label `1` for real or `0` for fake.
6. Split the dataset into training and test sets.
7. Train a Random Forest classifier.
8. Save the trained model to:

```text
models/note_model.joblib
```

The model used is:

```text
RandomForestClassifier
```

A Random Forest creates many decision trees. Each tree learns rules from the combined front+back feature values. During prediction, all trees vote, and the final result becomes `real` or `fake`.

## 5. Prediction Flow

When you capture or upload a note's front and back photo in the web page:

1. The browser waits until both photos are provided, then sends them to:

```text
POST /predict
```

as `front_file` and `back_file`.

2. `src/api.py` receives both images.
3. The same feature extraction process runs on both the front and back photo, and the two feature vectors are combined.
4. The serial number is read from the front photo using OCR.
5. The saved model is loaded from `models/note_model.joblib`.
6. The model predicts probabilities:

```text
real_probability
fake_probability
```

7. The web page displays:

```text
Model says real
```

or:

```text
Model says fake
```

along with confidence and the detected serial number.

## 6. How The Model Concludes Real Or Fake

The model does not use one single rule.

It combines all extracted features from both sides. For example:

- If the note has the expected shape on both sides,
- has similar denomination color on the front and back,
- has enough sharp print detail on both sides,
- has a visible thread-like pattern,
- has watermark-region variation,
- and looks similar to real examples from training,

then the model may predict `real`.

If the feature pattern — on either side, or in how the two sides relate to each other — is closer to images labelled as fake during training, it predicts `fake`.

The final output is based on learned statistical patterns, not legal proof.

## 7. Why Dataset Quality Matters

If you train with only a few images, the model can memorize those images.

For example, if all fake note pairs have a white background and all real note pairs have a dark background, the model may learn the background instead of the note.

Good training data should include:

- Different backgrounds.
- Different lighting.
- Different phones.
- Different angles.
- Different note conditions.
- Both front and back images for every note, correctly paired by filename.
- Photos not reused between training and testing.

## 8. Current Limitations

This model is a prototype.

It cannot fully verify physical security features such as:

- real watermark under transmitted light,
- actual raised intaglio print feel,
- optically variable ink movement,
- ultraviolet response,
- magnetic ink,
- paper fiber quality.

A phone photo can only capture visual clues, even with both sides photographed. For strong counterfeit detection, the system needs a larger labelled dataset of matched front/back pairs and possibly extra hardware or controlled lighting.

No dataset currently exists for this project, so no model has been trained yet.

## 9. Best Practical Use

Use this model as a learning and prototype system:

```text
Front photo + back photo -> extract visual features from each side -> combine -> trained classifier -> real/fake probability
```

Do not use it as final proof that a note is genuine or counterfeit.

For a college or demo project, explain that the model is a computer-vision classifier trained on labelled front/back image pairs and that accuracy depends on dataset size and quality.