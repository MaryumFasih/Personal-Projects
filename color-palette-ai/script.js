let colorData = {};
let themeKeys = [];
let themeEmbeddings = [];
let useModel;

// Load colors.json
fetch('colors.json')
  .then(res => res.json())
  .then(data => {
    colorData = data;
    themeKeys = Object.keys(colorData);
  })
  .catch(err => console.error('Failed to load colors.json:', err));

// Load USE & generate theme embeddings
use.load().then(model => {
  useModel = model;
  model.embed(themeKeys).then(embeddings => {
    themeEmbeddings = embeddings;
    console.log('Theme embeddings ready!');
  });
});

const generateBtn = document.getElementById('generateBtn');
const promptInput = document.getElementById('promptInput');
const paletteContainer = document.getElementById('paletteContainer');

generateBtn.addEventListener('click', async () => {
  const prompt = promptInput.value.trim().toLowerCase();
  paletteContainer.innerHTML = '';

  if (!prompt) {
    paletteContainer.innerHTML = '<p>Please enter a theme!</p>';
    return;
  }

  let matchedColors = null;

  // Exact match
  for (const keyword in colorData) {
    if (prompt.includes(keyword)) {
      matchedColors = colorData[keyword];
      console.log(`Exact match: "${keyword}"`);
      break;
    }
  }

  // Semantic match
  if (!matchedColors && useModel && themeEmbeddings) {
    const inputEmbedding = await useModel.embed([prompt]);
    const scores = await inputEmbedding.matMul(themeEmbeddings, false, true).array();
    const bestIndex = scores[0].indexOf(Math.max(...scores[0]));
    const bestTheme = themeKeys[bestIndex];
    matchedColors = colorData[bestTheme];
    console.log(`Semantic match: "${prompt}" → "${bestTheme}"`);
  }

  if (matchedColors) {
    showPalette(matchedColors);
  } else {
    const fallback = generatePaletteFromPrompt(prompt);
    showPalette(fallback);
  }
});

function showPalette(colors) {
  paletteContainer.innerHTML = '';
  colors.forEach(hex => {
    const wrapper = document.createElement('div');
    wrapper.className = 'swatch-wrapper';

    const swatch = document.createElement('div');
    swatch.className = 'swatch';
    swatch.style.backgroundColor = hex;

    const hexCode = document.createElement('div');
    hexCode.className = 'hexCode';
    hexCode.textContent = hex;

    swatch.addEventListener('click', () => {
      navigator.clipboard.writeText(hex);
      alert(`Copied ${hex}`);
    });

    wrapper.appendChild(swatch);
    wrapper.appendChild(hexCode);
    paletteContainer.appendChild(wrapper);
  });
}

// Fallback auto palette if no match
function generatePaletteFromPrompt(prompt) {
  const colors = [];
  const hash = hashCode(prompt);
  const baseHue = Math.abs(hash) % 360;

  for (let i = 0; i < 5; i++) {
    const hue = (baseHue + i * 30) % 360;
    const saturation = 65 + (i * 3) % 10;
    const lightness = 50 + (i * 4) % 10;
    colors.push(hslToHex(hue, saturation, lightness));
  }

  return colors;
}

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return hash;
}

function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n =>
    l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return `#${[f(0), f(8), f(4)]
    .map(x => Math.round(x * 255).toString(16).padStart(2, '0'))
    .join('')}`;
}
