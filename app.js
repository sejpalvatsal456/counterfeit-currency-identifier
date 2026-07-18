const noteProfiles = {
  10: { width: 123, height: 63, color: [116, 75, 54], features: ['Watermark', 'Security thread', 'See-through register', 'Micro lettering'] },
  20: { width: 129, height: 63, color: [185, 139, 68], features: ['Watermark', 'Security thread', 'See-through register', 'Micro lettering'] },
  50: { width: 135, height: 66, color: [75, 132, 125], features: ['Watermark', 'Security thread', 'See-through register', 'Micro lettering'] },
  100: { width: 142, height: 66, color: [128, 94, 151], features: ['Watermark', 'Security thread', 'See-through register', 'Latent image', 'Micro lettering'] },
  200: { width: 146, height: 66, color: [198, 151, 62], features: ['Watermark', 'Security thread', 'See-through register', 'Latent image', 'Micro lettering'] },
  500: { width: 150, height: 66, color: [112, 118, 92], features: ['Watermark', 'Windowed security thread', 'See-through register', 'Latent image', 'Micro lettering', 'Intaglio print'] },
  2000: { width: 166, height: 66, color: [167, 91, 143], features: ['Watermark', 'Windowed security thread', 'See-through register', 'Latent image', 'Micro lettering', 'Optically variable ink'] },
};

const photoInput = document.querySelector('#notePhoto');
const denominationInput = document.querySelector('#denomination');
const previewImage = document.querySelector('#previewImage');
const emptyPreview = document.querySelector('#emptyPreview');
const canvas = document.querySelector('#analysisCanvas');
const autoChecks = document.querySelector('#autoChecks');
const featureChecks = document.querySelector('#featureChecks');
const verdictText = document.querySelector('#verdictText');
const scoreText = document.querySelector('#scoreText');
const verdictNote = document.querySelector('#verdictNote');
const meterFill = document.querySelector('#meterFill');
const statusPill = document.querySelector('#statusPill');
const modelState = document.querySelector('#modelState');
const realProbability = document.querySelector('#realProbability');
const fakeProbability = document.querySelector('#fakeProbability');
const modelNote = document.querySelector('#modelNote');
const serialNumberStatus = document.querySelector('#serialNumberStatus');
const serialNumber = document.querySelector('#serialNumber');

let latestMetrics = null;
let latestFile = null;

function getApiBaseUrl() {
  if (window.location.port === '5500') {
    return `http://${window.location.hostname}:8000`;
  }
  return window.location.origin;
}

function init() {
  renderFeatureChecks();
  renderAutoChecks([]);
  photoInput.addEventListener('change', handlePhoto);
  denominationInput.addEventListener('change', () => {
    renderFeatureChecks();
    if (previewImage.src) analyzeImage();
    updateVerdict();
  });
}

function handlePhoto(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  latestFile = file;

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.style.display = 'block';
    emptyPreview.style.display = 'none';
    statusPill.textContent = 'Photo loaded';
    previewImage.onload = analyzeImage;
  };
  reader.readAsDataURL(file);
}

function analyzeImage() {
  const maxWidth = 900;
  const scale = Math.min(1, maxWidth / previewImage.naturalWidth);
  canvas.width = Math.max(1, Math.round(previewImage.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(previewImage.naturalHeight * scale));

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(previewImage, 0, 0, canvas.width, canvas.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  latestMetrics = getImageMetrics(imageData, canvas.width, canvas.height);

  const checks = buildAutoChecks(latestMetrics);
  renderAutoChecks(checks);
  updateVerdict();
  predictWithModel();
}

async function predictWithModel() {
  if (!latestFile) return;

  modelState.textContent = 'Checking';
  serialNumberStatus.textContent = 'Checking';
  modelNote.textContent = 'Sending the photo to the trained Python model...';
  realProbability.textContent = '--';
  fakeProbability.textContent = '--';
  serialNumber.textContent = '--';

  const formData = new FormData();
  formData.append('file', latestFile);
  formData.append('denomination', denominationInput.value);

  try {
    const predictUrl = `${getApiBaseUrl()}/predict`;
    const response = await fetch(predictUrl, {
      method: 'POST',
      body: formData,
    });
    const payload = await response.json();

    // console.log("Payload from /predict");
    // console.log(payload);

    if (!response.ok) {
      throw new Error(payload.detail || 'Model request failed.');
    }

    const confidence = Math.round(payload.confidence * 100);
    realProbability.textContent = `${Math.round(payload.real_probability * 100)}%`;
    fakeProbability.textContent = `${Math.round(payload.fake_probability * 100)}%`;
    modelState.textContent = payload.prediction === 'real' ? 'Real' : 'Fake';
    modelNote.textContent = `Model confidence: ${confidence}%. This result depends on the training dataset quality.`;
    serialNumber.textContent = payload.serial_number;
    serialNumberStatus.textContent = "Detected";

    verdictText.textContent = payload.prediction === 'real' ? 'Model says real' : 'Model says fake';
    scoreText.textContent = `${confidence}%`;
    meterFill.style.width = `${confidence}%`;
    meterFill.style.background = payload.prediction === 'real' ? 'var(--good)' : 'var(--danger)';
    statusPill.textContent = 'ML result';
  } catch (error) {
    modelState.textContent = 'Unavailable';
    modelNote.textContent = error.message;
  }
}

function getImageMetrics(imageData, width, height) {
  const data = imageData.data;
  let brightnessSum = 0;
  let redSum = 0;
  let greenSum = 0;
  let blueSum = 0;
  let sampleCount = 0;
  const step = 16;

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const index = (y * width + x) * 4;
      const red = data[index];
      const green = data[index + 1];
      const blue = data[index + 2];
      brightnessSum += 0.299 * red + 0.587 * green + 0.114 * blue;
      redSum += red;
      greenSum += green;
      blueSum += blue;
      sampleCount += 1;
    }
  }

  const blur = estimateSharpness(data, width, height);
  const brightness = brightnessSum / sampleCount;
  const averageColor = [redSum / sampleCount, greenSum / sampleCount, blueSum / sampleCount];
  const aspect = width / height;

  return { width, height, brightness, blur, averageColor, aspect };
}

function estimateSharpness(data, width, height) {
  let total = 0;
  let count = 0;
  const step = 6;

  for (let y = step; y < height - step; y += step) {
    for (let x = step; x < width - step; x += step) {
      const center = luminanceAt(data, width, x, y);
      const left = luminanceAt(data, width, x - step, y);
      const right = luminanceAt(data, width, x + step, y);
      const top = luminanceAt(data, width, x, y - step);
      const bottom = luminanceAt(data, width, x, y + step);
      total += Math.abs(4 * center - left - right - top - bottom);
      count += 1;
    }
  }

  return count ? total / count : 0;
}

function luminanceAt(data, width, x, y) {
  const index = (y * width + x) * 4;
  return 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
}

function buildAutoChecks(metrics) {
  const profile = getProfile();
  const noteAspect = profile.width / profile.height;
  const imageAspect = metrics.aspect;
  const aspectDelta = Math.abs(imageAspect - noteAspect) / noteAspect;
  const colorDistance = getColorDistance(metrics.averageColor, profile.color);

  return [
    {
      title: 'Photo brightness',
      value: `${Math.round(metrics.brightness)} / 255`,
      state: metrics.brightness >= 75 && metrics.brightness <= 215 ? 'pass' : 'warn',
      points: metrics.brightness >= 75 && metrics.brightness <= 215 ? 20 : 8,
    },
    {
      title: 'Photo sharpness',
      value: metrics.blur.toFixed(1),
      state: metrics.blur >= 10 ? 'pass' : 'warn',
      points: metrics.blur >= 10 ? 20 : 6,
    },
    {
      title: 'Expected note shape',
      value: `${imageAspect.toFixed(2)} vs ${noteAspect.toFixed(2)}`,
      state: aspectDelta < 0.28 ? 'pass' : 'warn',
      points: aspectDelta < 0.28 ? 18 : 7,
    },
    {
      title: 'Denomination color hint',
      value: colorDistance < 110 ? 'close' : 'different',
      state: colorDistance < 110 ? 'pass' : 'warn',
      points: colorDistance < 110 ? 14 : 5,
    },
  ];
}

function renderAutoChecks(checks) {
  autoChecks.innerHTML = '';
  const rows = checks.length
    ? checks
    : [
        { title: 'Photo brightness', value: 'waiting', state: 'warn' },
        { title: 'Photo sharpness', value: 'waiting', state: 'warn' },
        { title: 'Expected note shape', value: 'waiting', state: 'warn' },
        { title: 'Denomination color hint', value: 'waiting', state: 'warn' },
      ];

  rows.forEach((check) => {
    const row = document.createElement('div');
    row.className = 'check-row';
    row.dataset.points = check.points || 0;
    row.innerHTML = `
      <span class="dot ${check.state}"></span>
      <span class="check-title">${check.title}</span>
      <span class="check-value">${check.value}</span>
    `;
    autoChecks.appendChild(row);
  });
}

function renderFeatureChecks() {
  featureChecks.innerHTML = '';
  getProfile().features.forEach((feature) => {
    const label = document.createElement('label');
    label.className = 'feature-toggle';
    label.innerHTML = `<input type="checkbox" value="${feature}" /> <span>${feature}</span>`;
    label.querySelector('input').addEventListener('change', updateVerdict);
    featureChecks.appendChild(label);
  });
}

function updateVerdict() {
  const autoPoints = [...autoChecks.querySelectorAll('.check-row')].reduce(
    (sum, row) => sum + Number(row.dataset.points || 0),
    0
  );
  const featureInputs = [...featureChecks.querySelectorAll('input')];
  const selectedFeatures = featureInputs.filter((input) => input.checked).length;
  const featurePoints = featureInputs.length ? (selectedFeatures / featureInputs.length) * 28 : 0;
  const score = Math.round(Math.min(100, autoPoints + featurePoints));

  scoreText.textContent = `${score}%`;
  meterFill.style.width = `${score}%`;

  if (!latestMetrics) {
    verdictText.textContent = 'Waiting for photo';
    verdictNote.textContent = 'Add a clear photo, then mark visible security features below.';
    meterFill.style.background = 'var(--danger)';
    statusPill.textContent = 'No photo';
    return;
  }

  if (score >= 78) {
    verdictText.textContent = 'Likely genuine';
    verdictNote.textContent = 'The photo checks and selected security features look consistent for a prototype review.';
    meterFill.style.background = 'var(--good)';
    statusPill.textContent = 'Strong match';
  } else if (score >= 52) {
    verdictText.textContent = 'Needs inspection';
    verdictNote.textContent = 'Some checks are inconclusive. Retake the photo in brighter light and inspect the note physically.';
    meterFill.style.background = 'var(--amber)';
    statusPill.textContent = 'Review needed';
  } else {
    verdictText.textContent = 'Suspicious';
    verdictNote.textContent = 'The note did not match enough visible checks. Do not rely on this alone for a final decision.';
    meterFill.style.background = 'var(--danger)';
    statusPill.textContent = 'Low match';
  }
}

function getColorDistance(colorA, colorB) {
  return Math.sqrt(
    (colorA[0] - colorB[0]) ** 2 + (colorA[1] - colorB[1]) ** 2 + (colorA[2] - colorB[2]) ** 2
  );
}

function getProfile() {
  return noteProfiles[denominationInput.value];
}

init();
