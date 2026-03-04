/**
 * DNSMOS (Deep Noise Suppression Mean Opinion Score) — browser inference via ONNX Runtime Web.
 *
 * Two ONNX models:
 *   1. model_v8.onnx  (p808 MOS)  — input: mel spectrogram [N, 900, 120]  → output [N, 1]
 *   2. sig_bak_ovr.onnx           — input: raw audio [N, 144160]           → output [N, 3]
 *
 * Combined output after polyfit: [p808_mos, mos_sig, mos_bak, mos_ovr]
 */

import * as ort from "onnxruntime-web";

// Tell onnxruntime-web where to find its WASM files (served from /public)
ort.env.wasm.wasmPaths = "/";

// Suppress "Unknown CPU vendor" warning on Apple Silicon — purely cosmetic,
// onnxruntime WASM doesn't recognise ARM vendors but inference is unaffected.
ort.env.logLevel = "error";

// ─── Constants ───────────────────────────────────────────────────────────────
const SAMPLING_RATE = 16000;
const INPUT_LENGTH = 9.01; // seconds — fixed model window size
const LEN_SAMPLES = Math.round(INPUT_LENGTH * SAMPLING_RATE); // 144160
const MAX_DURATION = 50; // seconds — maximum accepted input length
const MAX_SAMPLES = MAX_DURATION * SAMPLING_RATE;

// Mel spectrogram parameters (matches Python DNSMOS defaults)
const N_MELS = 120;
const N_FFT = 321; // frame_size + 1 = 320 + 1
const HOP_LENGTH = 160;
const FRAME_SIZE = 320;

// ─── Polyfit coefficients (non-personalized) ─────────────────────────────────
function polyfitSig(x: number): number {
  return -0.08397278 * x * x + 1.22083953 * x + 0.0052439;
}
function polyfitBak(x: number): number {
  return -0.13166888 * x * x + 1.60915514 * x - 0.39604546;
}
function polyfitOvr(x: number): number {
  return -0.06766283 * x * x + 1.11546468 * x + 0.04602535;
}

// ─── Mel filterbank (computed once) ──────────────────────────────────────────
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

function createMelFilterbank(
  sr: number,
  nFft: number,
  nMels: number,
): Float32Array[] {
  const fMin = 0;
  const fMax = sr / 2;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const nFreqs = Math.floor(nFft / 2) + 1;

  // nMels + 2 equally spaced mel points
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = melMin + ((melMax - melMin) * i) / (nMels + 1);
  }

  const freqPoints = melPoints.map((m) => melToHz(m));
  const binPoints = freqPoints.map((f) =>
    Math.floor(((nFft - 1 + 1) * f) / sr),
  );

  const filters: Float32Array[] = [];
  for (let m = 0; m < nMels; m++) {
    const filter = new Float32Array(nFreqs);
    const start = binPoints[m];
    const center = binPoints[m + 1];
    const end = binPoints[m + 2];

    for (let k = start; k <= center; k++) {
      filter[k] = center === start ? 0 : (k - start) / (center - start);
    }
    for (let k = center; k <= end; k++) {
      filter[k] = end === center ? 0 : (end - k) / (end - center);
    }

    // Slaney normalization
    const enorm = 2.0 / (melToHz(melPoints[m + 2]) - melToHz(melPoints[m]));
    for (let k = 0; k < nFreqs; k++) {
      filter[k] *= enorm;
    }
    filters.push(filter);
  }
  return filters;
}

// ─── STFT (real-valued, Hann window) ─────────────────────────────────────────
function hannWindow(size: number): Float32Array {
  const w = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / size));
  }
  return w;
}

function stftMagnitude(
  signal: Float32Array,
  nFft: number,
  hopLength: number,
  winLength: number,
): Float32Array[] {
  // Center-pad signal like librosa (reflect padding)
  const padLen = Math.floor(nFft / 2);
  const padded = new Float32Array(signal.length + 2 * padLen);
  // Reflect left
  for (let i = 0; i < padLen; i++) {
    padded[i] = signal[padLen - i];
  }
  // Copy signal
  padded.set(signal, padLen);
  // Reflect right
  for (let i = 0; i < padLen; i++) {
    padded[padLen + signal.length + i] =
      signal[signal.length - 2 - i] ?? signal[signal.length - 1];
  }

  const win = hannWindow(winLength);
  const nFrames = 1 + Math.floor((padded.length - nFft) / hopLength);
  const nFreqs = Math.floor(nFft / 2) + 1;
  const frames: Float32Array[] = [];

  for (let t = 0; t < nFrames; t++) {
    const offset = t * hopLength;
    // Apply window and compute DFT magnitudes
    const magnitudes = new Float32Array(nFreqs);
    for (let k = 0; k < nFreqs; k++) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < winLength; n++) {
        const val = padded[offset + n] * win[n];
        const angle = (2 * Math.PI * k * n) / nFft;
        re += val * Math.cos(angle);
        im -= val * Math.sin(angle);
      }
      magnitudes[k] = Math.sqrt(re * re + im * im);
    }
    frames.push(magnitudes);
  }
  return frames; // nFrames arrays of nFreqs each
}

// ─── Mel spectrogram (power → dB, normalized) ───────────────────────────────
let cachedFilterbank: Float32Array[] | null = null;

function computeMelSpectrogram(audio: Float32Array): Float32Array {
  if (!cachedFilterbank) {
    cachedFilterbank = createMelFilterbank(SAMPLING_RATE, N_FFT, N_MELS);
  }
  const fb = cachedFilterbank;
  const nFreqs = Math.floor(N_FFT / 2) + 1;

  // STFT magnitude
  const mag = stftMagnitude(audio, N_FFT, HOP_LENGTH, FRAME_SIZE);
  const nFrames = mag.length;

  // Power spectrum → mel
  const melSpec = new Float32Array(nFrames * N_MELS);
  for (let t = 0; t < nFrames; t++) {
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      for (let k = 0; k < nFreqs; k++) {
        sum += mag[t][k] * mag[t][k] * fb[m][k]; // power * filter
      }
      melSpec[t * N_MELS + m] = sum;
    }
  }

  // Power to dB (ref = max, top_db = 80) + normalize to [-1, 1] range
  let maxVal = -Infinity;
  for (let i = 0; i < melSpec.length; i++) {
    if (melSpec[i] > maxVal) maxVal = melSpec[i];
  }
  const refDb = 10 * Math.log10(Math.max(maxVal, 1e-10));
  for (let i = 0; i < melSpec.length; i++) {
    let db = 10 * Math.log10(Math.max(melSpec[i], 1e-10)) - refDb;
    db = Math.max(db, -80);
    melSpec[i] = (db + 40) / 40; // matches Python: (power_to_db(ref=max) + 40) / 40
  }

  return melSpec; // flat [nFrames, N_MELS]
}

// ─── Session cache ───────────────────────────────────────────────────────────
let sigBakOvrSession: ort.InferenceSession | null = null;
let p808Session: ort.InferenceSession | null = null;

async function getSigBakOvrSession(): Promise<ort.InferenceSession> {
  if (!sigBakOvrSession) {
    sigBakOvrSession = await ort.InferenceSession.create(
      "/models/sig_bak_ovr.onnx",
      { executionProviders: ["wasm"] },
    );
  }
  return sigBakOvrSession;
}

async function getP808Session(): Promise<ort.InferenceSession> {
  if (!p808Session) {
    p808Session = await ort.InferenceSession.create("/models/model_v8.onnx", {
      executionProviders: ["wasm"],
    });
  }
  return p808Session;
}

// ─── Resample (linear interpolation, good enough for 48→16 kHz) ─────────────
function resample(audio: Float32Array, fromSr: number, toSr: number): Float32Array {
  if (fromSr === toSr) return audio;
  const ratio = toSr / fromSr;
  const newLen = Math.round(audio.length * ratio);
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const srcIdx = i / ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, audio.length - 1);
    const frac = srcIdx - lo;
    out[i] = audio[lo] * (1 - frac) + audio[hi] * frac;
  }
  return out;
}

// ─── Public API ──────────────────────────────────────────────────────────────
export interface DNSMOSResult {
  /** P.808 overall MOS prediction */
  p808_mos: number;
  /** Signal quality */
  mos_sig: number;
  /** Background noise quality */
  mos_bak: number;
  /** Overall quality */
  mos_ovr: number;
}

/**
 * Run DNSMOS inference on raw audio samples.
 *
 * @param audioData  PCM float32 samples (mono, any sample rate)
 * @param sampleRate Sample rate of the input audio
 * @returns DNSMOS scores
 */
export async function runDNSMOS(
  audioData: Float32Array,
  sampleRate: number,
): Promise<DNSMOSResult> {
  // Resample to 16 kHz if needed
  let audio =
    sampleRate !== SAMPLING_RATE
      ? resample(audioData, sampleRate, SAMPLING_RATE)
      : audioData;

  // Enforce max duration
  if (audio.length > MAX_SAMPLES) {
    throw new Error(
      `Audio is ${(audio.length / SAMPLING_RATE).toFixed(1)}s — max is ${MAX_DURATION}s. Please trim your file.`,
    );
  }

  // Pad/repeat to minimum model window length
  while (audio.length < LEN_SAMPLES) {
    const combined = new Float32Array(audio.length * 2);
    combined.set(audio);
    combined.set(audio, audio.length);
    audio = combined;
  }

  // Score the entire audio as one piece.
  // The model has a fixed 9.01s input window, so we tile non-overlapping
  // windows across the full duration and average the results.
  const numWindows = Math.max(1, Math.floor(audio.length / LEN_SAMPLES));

  const allScores: number[][] = [];

  const [sigSess, p808Sess] = await Promise.all([
    getSigBakOvrSession(),
    getP808Session(),
  ]);

  for (let idx = 0; idx < numWindows; idx++) {
    const start = idx * LEN_SAMPLES;
    const end = start + LEN_SAMPLES;
    if (end > audio.length) break;

    const segment = audio.slice(start, end);

    // ── sig_bak_ovr model: raw audio [1, 144160] ──
    const sigInput = new ort.Tensor("float32", segment, [1, LEN_SAMPLES]);
    const sigResult = await sigSess.run({ input_1: sigInput });
    const sigOut = sigResult["Identity:0"]?.data as Float32Array ??
      sigResult[Object.keys(sigResult)[0]]?.data as Float32Array;

    // ── p808 model: mel spectrogram [1, 900, 120] ──
    // Trim last 160 samples (matches Python: audio_seg[..., :-160])
    const trimmed = segment.slice(0, segment.length - 160);
    const melFlat = computeMelSpectrogram(trimmed);
    // melFlat is [nFrames, 120]; we need exactly [900, 120]
    const nFrames = melFlat.length / N_MELS;
    const targetFrames = 900;
    const melPadded = new Float32Array(targetFrames * N_MELS);
    const copyFrames = Math.min(nFrames, targetFrames);
    melPadded.set(melFlat.subarray(0, copyFrames * N_MELS));

    const p808Input = new ort.Tensor("float32", melPadded, [
      1,
      targetFrames,
      N_MELS,
    ]);
    const p808Result = await p808Sess.run({ input_1: p808Input });
    const p808Out = p808Result["Identity:0"]?.data as Float32Array ??
      p808Result[Object.keys(p808Result)[0]]?.data as Float32Array;

    // Combine: [p808_mos, sig, bak, ovr]
    allScores.push([p808Out[0], sigOut[0], sigOut[1], sigOut[2]]);
  }

  // Average over segments
  const avg = [0, 0, 0, 0];
  for (const s of allScores) {
    for (let i = 0; i < 4; i++) avg[i] += s[i];
  }
  for (let i = 0; i < 4; i++) avg[i] /= allScores.length;

  // Apply polyfit
  return {
    p808_mos: avg[0],
    mos_sig: polyfitSig(avg[1]),
    mos_bak: polyfitBak(avg[2]),
    mos_ovr: polyfitOvr(avg[3]),
  };
}
