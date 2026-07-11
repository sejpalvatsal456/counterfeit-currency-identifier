# How The ML Model Works

This project detects whether an Indian currency note photo is likely `real` or `fake` using a supervised machine-learning model.

It does not magically know fake notes by itself. It learns patterns from labelled examples placed in:

```text
dataset/real/
dataset/fake/
```

## 1. Training Data

You first collect photos of notes and label them:

- `dataset/real/` contains genuine note photos.
- `dataset/fake/` contains fake, counterfeit, or approved fake-like note photos.

For best results, train one denomination at a time, for example only INR 500 notes.

The model compares the visual patterns in real-note images against fake-note images. If the dataset is small, the model may appear accurate but fail on new photos.

## 2. Feature Extraction

The file `src/features.py` extracts numeric features from every note photo.

When an image is loaded, the code tries to find the largest note-like rectangular region. It uses OpenCV edge detection and contours to crop the note area. Then it resizes the note image to a fixed size so all training images are measured consistently.

After that, it extracts features such as:

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

These are not final decisions. They are measurements that describe the note image.

## 3. Important Features

### Aspect Ratio

Indian notes have expected width and height ratios. The model checks whether the photographed note shape is close to the selected denomination.

Example:

```text
INR 500 expected ratio = 150 / 66
```

If the photo has a very different shape, it may indicate poor cropping, wrong denomination, or a fake-like sample.

### Brightness And Contrast

The model measures whether the note is too dark, too bright, or has enough visible print contrast.

Bad lighting can reduce accuracy, so the model needs training photos from different lighting conditions.

### Sharpness

Sharpness is calculated using the Laplacian variance method. A blurry image has fewer high-frequency details.

This matters because security features such as micro text, borders, and fine print are harder to detect in blurry images.

### Edge Density

Currency notes contain many printed lines, borders, symbols, numbers, and texture patterns. Edge density measures how much visual structure exists in the image.

Fake-like notes may have different print detail or lower-quality edges.

### Security Thread Score

The code looks for strong vertical structures in the central note region. This is a rough computer-vision estimate for a visible security thread.

It does not truly verify the physical thread. It only measures whether the image contains a thread-like vertical pattern.

### Watermark Score

The code checks the left-side watermark region for tonal variation and local contrast.

A real watermark is a physical feature seen by light passing through the note. A normal phone photo cannot always verify it correctly, so this score is only a visual hint.

### See-Through Register Score

The code looks at a region where see-through register patterns may appear and measures edge density.

This is also a rough visual feature, not a certified security test.

### Micro Text Score

The model measures high-frequency details in lower printed regions. Micro lettering and fine print create small sharp patterns.

If the note image is blurry, this score may be low even for a real note.

### Dominant Color Distance

Each denomination has an expected color profile. The model compares the average photo color with the expected color for the selected denomination.

For example, INR 500 has a different color profile than INR 200 or INR 100.

Color alone is not enough because camera lighting can change it, but it helps as one feature.

## 4. Training The Classifier

The file `src/train_model.py` loads all images from the dataset folders.

For each image:

1. Read the image using OpenCV.
2. Extract feature values using `extract_features`.
3. Assign label `1` for real or `0` for fake.
4. Split the dataset into training and test sets.
5. Train a Random Forest classifier.
6. Save the trained model to:

```text
models/note_model.joblib
```

The model used is:

```text
RandomForestClassifier
```

A Random Forest creates many decision trees. Each tree learns rules from the feature values. During prediction, all trees vote, and the final result becomes `real` or `fake`.

## 5. Prediction Flow

When you upload or click a note photo in the web page:

1. The browser sends the image to:

```text
POST /predict
```

2. `src/api.py` receives the image.
3. The same feature extraction process runs on the uploaded image.
4. The saved model is loaded from `models/note_model.joblib`.
5. The model predicts probabilities:

```text
real_probability
fake_probability
```

6. The web page displays:

```text
Model says real
```

or:

```text
Model says fake
```

along with confidence.

## 6. How The Model Concludes Real Or Fake

The model does not use one single rule.

It combines all extracted features. For example:

- If the note has the expected shape,
- has similar denomination color,
- has enough sharp print detail,
- has a visible thread-like pattern,
- has watermark-region variation,
- and looks similar to real examples from training,

then the model may predict `real`.

If the feature pattern is closer to images labelled as fake during training, it predicts `fake`.

The final output is based on learned statistical patterns, not legal proof.

## 7. Why Dataset Quality Matters

If you train with only a few images, the model can memorize those images.

For example, if all fake images have a white background and all real images have a dark background, the model may learn the background instead of the note.

Good training data should include:

- Different backgrounds.
- Different lighting.
- Different phones.
- Different angles.
- Different note conditions.
- Front and back images.
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

A phone photo can only capture visual clues. For strong counterfeit detection, the system needs a larger labelled dataset and possibly extra hardware or controlled lighting.

## 9. Best Practical Use

Use this model as a learning and prototype system:

```text
Input photo -> extract visual features -> trained classifier -> real/fake probability
```

Do not use it as final proof that a note is genuine or counterfeit.

For a college or demo project, explain that the model is a computer-vision classifier trained on labelled images and that accuracy depends on dataset size and quality.
