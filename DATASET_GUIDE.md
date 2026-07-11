# Dataset Guide

Public datasets with real counterfeit Indian banknote images are difficult to find. Most public currency datasets are for denomination recognition, not fake-vs-real authentication.

## Best Practical Sources

1. Genuine notes you photograph yourself.
   - Use your own legal banknotes.
   - Capture front and back.
   - Use different phones, distances, backgrounds, lighting, tilt, and worn/new notes.
   - Put these in `dataset/real/`.

2. Institution-approved counterfeit samples.
   - Ask a bank, college lab, forensic lab, or police/cybercrime awareness cell whether they can provide supervised access to counterfeit-note images.
   - Do not buy, print, circulate, or keep counterfeit notes.
   - Put approved images in `dataset/fake/`.

3. Public currency-recognition datasets.
   - Useful for pretraining denomination detection.
   - Usually not enough for fake-vs-real detection because they mostly contain genuine notes.

4. Controlled fake-like negatives.
   - For demos only, you can use legal specimen/play-money/training-note images if they are clearly not real currency.
   - This helps test the pipeline, but it does not prove the model can detect real counterfeits.

## Recommended Collection Plan

Start with one denomination, usually INR 500.

```text
dataset/
  real/
    500_real_001_front.jpg
    500_real_001_back.jpg
  fake/
    500_fake_001_front.jpg
    500_fake_001_back.jpg
```

Minimum useful target:

- 100 genuine images
- 100 fake or approved fake-like images
- 20 percent held out for testing

Better target:

- 500+ genuine images
- 500+ fake images
- Photos from phones that are not used in training

## Do Not

- Do not create counterfeit notes.
- Do not download random note photos and assume they are fake.
- Do not mix old and new note series in the same model unless you label the series.
- Do not train on only internet images and expect phone-camera performance.

## Notes

If you cannot get real counterfeit samples, change the project goal first:

- Phase 1: denomination and note-region detection.
- Phase 2: security-feature visibility scoring.
- Phase 3: fake-vs-real only after approved labelled counterfeit data is available.
