# Dataset Guide

Public datasets with real counterfeit Indian banknote images are difficult to find. Most public currency datasets are for denomination recognition, not fake-vs-real authentication.

The model now looks at **both sides of the note together**, so every sample needs a matched front photo and back photo. A note with only one side photographed cannot be used for training.

## Best Practical Sources

1. Genuine notes you photograph yourself.
   - Use your own legal banknotes.
   - Photograph both the front and the back of every note.
   - Use different phones, distances, backgrounds, lighting, tilt, and worn/new notes.
   - Put these in `dataset/real/front/` and `dataset/real/back/`.

2. Institution-approved counterfeit samples.
   - Ask a bank, college lab, forensic lab, or police/cybercrime awareness cell whether they can provide supervised access to counterfeit-note images.
   - Do not buy, print, circulate, or keep counterfeit notes.
   - Get front and back photos of each approved sample if possible.
   - Put approved images in `dataset/fake/front/` and `dataset/fake/back/`.

3. Public currency-recognition datasets.
   - Useful for pretraining denomination detection.
   - Usually not enough for fake-vs-real detection because they mostly contain genuine notes, and many only show the front.

4. Controlled fake-like negatives.
   - For demos only, you can use legal specimen/play-money/training-note images if they are clearly not real currency.
   - This helps test the pipeline, but it does not prove the model can detect real counterfeits.

## Recommended Collection Plan

Start with one denomination, usually INR 500.

Each note needs a front image and a back image that share the same filename (stem), so the training script can pair them automatically:

```text
dataset/
  real/
    front/
      500_real_001.jpg
      500_real_002.jpg
    back/
      500_real_001.jpg
      500_real_002.jpg
  fake/
    front/
      500_fake_001.jpg
    back/
      500_fake_001.jpg
```

`front/500_real_001.jpg` and `back/500_real_001.jpg` are treated as the two sides of the same physical note. If a front image has no matching back image (or vice versa), the training script skips it and prints a warning rather than guessing.

Minimum useful target:

- 100 genuine notes (100 front + 100 back photos)
- 100 fake or approved fake-like notes (100 front + 100 back photos)
- 20 percent held out for testing

Better target:

- 500+ genuine notes, front and back
- 500+ fake notes, front and back
- Photos from phones that are not used in training

## Do Not

- Do not create counterfeit notes.
- Do not download random note photos and assume they are fake.
- Do not mix old and new note series in the same model unless you label the series.
- Do not train on only internet images and expect phone-camera performance.
- Do not leave a note with only a front or only a back photo in the dataset; unpaired images are skipped and wasted.

## Notes

If you cannot get real counterfeit samples, change the project goal first:

- Phase 1: denomination and note-region detection.
- Phase 2: security-feature visibility scoring.
- Phase 3: fake-vs-real only after approved labelled counterfeit data is available (front and back).

## Current Status

No dataset has been collected yet. `dataset/real/front`, `dataset/real/back`, `dataset/fake/front`, and `dataset/fake/back` need to be created and populated with matched front/back photo pairs before `python -m src.train_model` can produce a usable model.