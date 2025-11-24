/* app.js — uses Web Audio AnalyserNode for spectrum (no external FFT lib) */
let audioCtx, micNode, analyser, scriptNode;
let started = false;
let sampleRate = 44100;
let fftSize = 2048; // analyser FFT size (power of two)
const waveformCanvas = document.getElementById('waveform');
const spectrumCanvas = document.getElementById('spectrum');
const spectrogramCanvas = document.getElementById('spectrogram');
const wfCtx = waveformCanvas.getContext('2d');
const spCtx = spectrumCanvas.getContext('2d');
const specCtx = spectrogramCanvas.getContext('2d');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const sampleRateSelect = document.getElementById('sampleRate');
const muSlider = document.getElementById('mu');
const muVal = document.getElementById('muVal');
const orderSlider = document.getElementById('filterOrder');
const orderVal = document.getElementById('orderVal');
const firPreset = document.getElementById('firPreset');

const snrBeforeEl = document.getElementById('snrBefore');
const snrAfterEl = document.getElementById('snrAfter');
const mseEl = document.getElementById('mse');

muSlider.oninput = () => muVal.textContent = muSlider.value;
orderSlider.oninput = () => orderVal.textContent = orderSlider.value;
sampleRateSelect.onchange = () => { sampleRate = parseInt(sampleRateSelect.value, 10); };

startBtn.onclick = start;
stopBtn.onclick = stop;

let freqData, timeData;

function start(){
  if (started) return;
  started = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate});
    const src = audioCtx.createMediaStreamSource(stream);

    // analyser for visualization
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = fftSize;
    analyser.smoothingTimeConstant = 0.2;

    // ScriptProcessor used for LMS and processing; small buffer
    const bufferSize = 1024;
    scriptNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);

    src.connect(analyser);
    analyser.connect(scriptNode);
    scriptNode.connect(audioCtx.destination);

    micNode = src;

    // allocate arrays
    const bins = analyser.frequencyBinCount;
    freqData = new Float32Array(bins);
    timeData = new Float32Array(analyser.fftSize);

    // LMS init
    setupLMS(+orderSlider.value);

    scriptNode.onaudioprocess = audioProcess;
  }).catch(err => {
    alert('Microphone access denied or not available: ' + err.message);
    console.error(err);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
  });
}

function stop(){
  if (!started) return;
  started = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  if (scriptNode) { scriptNode.disconnect(); scriptNode.onaudioprocess = null; scriptNode = null; }
  if (analyser) { analyser.disconnect(); analyser = null; }
  if (micNode) { micNode.disconnect(); micNode = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
}

/* ========== Visualization using analyser data ========== */
function drawWaveform(floatBuf) {
  const w = waveformCanvas.width, h = waveformCanvas.height;
  wfCtx.clearRect(0,0,w,h);
  wfCtx.beginPath();
  wfCtx.moveTo(0, h/2);
  const step = Math.max(1, Math.floor(floatBuf.length / w));
  for (let i=0;i<w;i++){
    const idx = i*step;
    const v = floatBuf[idx] || 0;
    const y = (1 - (v+1)/2) * h;
    wfCtx.lineTo(i, y);
  }
  wfCtx.strokeStyle = '#79ffe1';
  wfCtx.lineWidth = 1.5;
  wfCtx.stroke();
}

function drawSpectrumFromAnalyser() {
  const w = spectrumCanvas.width, h = spectrumCanvas.height;
  spCtx.clearRect(0,0,w,h);
  analyser.getFloatFrequencyData(freqData); // fills freqData in dB (negative values)
  const N = freqData.length;
  spCtx.beginPath();
  for (let i=0;i<N;i++){
    const x = Math.floor(i/N * w);
    // freqData are in dBFS, typically between -140 and 0
    const db = freqData[i];
    const y = h - ((db + 140) / 140) * h; // map [-140,0] to [h,0]
    spCtx.lineTo(x, y);
  }
  spCtx.strokeStyle = '#ffd166';
  spCtx.lineWidth = 1.2;
  spCtx.stroke();
}

/* simple scrolling spectrogram using analyser */
function drawSpectrogramColumnFromAnalyser() {
  const w = spectrogramCanvas.width, h = spectrogramCanvas.height;
  // scroll left
  const imageData = specCtx.getImageData(2,0,w-2,h);
  specCtx.putImageData(imageData, 0, 0);
  specCtx.clearRect(w-2,0,2,h);

  analyser.getFloatFrequencyData(freqData);
  const N = freqData.length;
  for (let i=0;i<N;i++){
    const row = Math.floor((i/N) * h);
    const db = freqData[i]; // negative values
    const intensity = Math.max(0, Math.min(255, Math.floor( ((db + 140) / 140) * 255 )));
    specCtx.fillStyle = `rgba(${intensity}, ${Math.floor(intensity*0.7)}, 40, 1)`;
    specCtx.fillRect(w-2, h-row, 2, 2);
  }
}

/* ========== LMS adaptive filter ========== */
let lmsWeights = [];
let lmsOrder = 32;
let lmsX = [];

function setupLMS(order) {
  lmsOrder = order;
  lmsWeights = new Float32Array(lmsOrder).fill(0);
  lmsX = new Float32Array(lmsOrder).fill(0);
}

function lmsStep(xSample, desired) {
  for (let i=lmsOrder-1;i>0;i--) lmsX[i] = lmsX[i-1];
  lmsX[0] = xSample;

  let y = 0;
  for (let i=0;i<lmsOrder;i++) y += lmsWeights[i] * lmsX[i];

  const e = desired - y;
  const mu = parseFloat(muSlider.value);
  for (let i=0;i<lmsOrder;i++) {
    lmsWeights[i] += mu * e * lmsX[i];
  }
  return {y, e};
}

function computeSNR(signal, noise) {
  let ps = 0, pn = 0;
  for (let i=0;i<signal.length;i++){
    ps += signal[i]*signal[i];
    pn += noise[i]*noise[i];
  }
  ps = ps / signal.length || 1e-12;
  pn = pn / signal.length || 1e-12;
  const snr = 10 * Math.log10(ps / pn);
  return snr;
}

/* main audio process */
function audioProcess(evt) {
  const inBuf = evt.inputBuffer.getChannelData(0);
  // copy to snapshot for display/processing
  const snapshot = new Float32Array(inBuf.length);
  snapshot.set(inBuf);

  // draw waveform using time domain data from analyser for consistent buffer sizes
  analyser.getFloatTimeDomainData(timeData);
  drawWaveform(timeData.subarray(0, Math.min(timeData.length, snapshot.length)));

  // spectrum and spectrogram from analyser
  drawSpectrumFromAnalyser();
  drawSpectrogramColumnFromAnalyser();

  // Adaptive filter demo: create a reference signal as a delayed, scaled version of input
  const reference = new Float32Array(snapshot.length);
  const delay = 5;
  for (let i=0;i<snapshot.length;i++) reference[i] = snapshot[Math.max(0, i - delay)] * 0.9;

  const errorArr = new Float32Array(snapshot.length);
  const outArr = new Float32Array(snapshot.length);
  for (let n=0; n<snapshot.length; n++){
    const x = reference[n];
    const d = snapshot[n];
    const {y, e} = lmsStep(x, d);
    outArr[n] = d - y;
    errorArr[n] = e;
  }

  // compute MSE and quick SNR estimates
  let mse = 0;
  for (let i=0;i<errorArr.length;i++) mse += errorArr[i]*errorArr[i];
  mse /= errorArr.length;
  mseEl.textContent = mse.toFixed(6);

  const snrAfter = computeSNR(outArr, errorArr);
  const snrBefore = computeSNR(snapshot, reference);
  snrBeforeEl.textContent = snrBefore.toFixed(2) + ' dB';
  snrAfterEl.textContent = snrAfter.toFixed(2) + ' dB';
}

muSlider.addEventListener('input', () => {});
orderSlider.addEventListener('change', () => { setupLMS(parseInt(orderSlider.value, 10)); });
firPreset.addEventListener('change', () => { console.log('FIR preset:', firPreset.value); });
