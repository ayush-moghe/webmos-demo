# webmos-demo — Browser Audio Quality Scorer

Client-side audio quality scoring powered by the [webmos](https://www.npmjs.com/package/webmos) npm package, which runs Microsoft's [DNSMOS](https://arxiv.org/abs/2010.15258) (Deep Noise Suppression Mean Opinion Score) model entirely in the browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). No server-side inference needed — zero compute cost.

![App screenshot](example.png)

## What it does

Drop or select a WAV/MP3 file and get MOS scores instantly:

| Score | Description |
|-------|-------------|
| **SIG** | Speech signal quality |
| **BAK** | Background noise quality |
| **OVR** | Overall speech + noise quality |

All inference runs in-browser using WebAssembly — nothing leaves your machine.

## How it works

1. Audio is decoded to 16 kHz mono PCM via the Web Audio API
2. The [webmos](https://www.npmjs.com/package/webmos) package handles everything else — model loading, inference, and score calibration

## Project structure

```
mos-app/
├── app/
│   ├── page.tsx              # Drag-and-drop UI
│   ├── layout.tsx
│   └── globals.css
```

## Getting started

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and drop an audio file.

## Tech stack

- **Next.js 16** (Turbopack) + React 19
- **[webmos](https://www.npmjs.com/package/webmos)** — DNSMOS inference in the browser
- **Tailwind CSS 4** — styling
- **TypeScript** — end to end
