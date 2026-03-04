"use client";

import { useState, useRef, useCallback } from "react";
import type { DNSMOSResult } from "webmos";

export default function Home() {
  const [result, setResult] = useState<DNSMOSResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setFileName(file.name);

    try {
      // Lazy-load the webmos package (large WASM init)
      const { runDNSMOS } = await import("webmos");

      // Decode audio file
      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

      // Take first channel, mono
      const pcm = audioBuffer.getChannelData(0);
      const sampleRate = audioBuffer.sampleRate;

      const scores = await runDNSMOS(pcm, sampleRate);
      setResult(scores);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-8 p-8 bg-gray-950 text-gray-100">
      <h1 className="text-3xl font-bold">DNSMOS — Browser Audio Quality</h1>
      <p className="text-gray-400 max-w-lg text-center">
        Drop or select a WAV/MP3 file to get MOS scores entirely in your browser
        using ONNX Runtime Web. No server needed.
      </p>

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => fileRef.current?.click()}
        className="border-2 border-dashed border-gray-600 rounded-xl w-full max-w-md h-40 flex items-center justify-center cursor-pointer hover:border-blue-500 transition-colors"
      >
        <span className="text-gray-400">
          {loading
            ? "Processing…"
            : fileName
              ? fileName
              : "Drop audio file here or click to browse"}
        </span>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      {/* Results */}
      {result && (
        <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md space-y-3">
          <h2 className="text-lg font-semibold mb-2">Results</h2>
          <ScoreRow label="Signal (SIG)" value={result.mos_sig} />
          <ScoreRow label="Background (BAK)" value={result.mos_bak} />
          <ScoreRow label="Overall (OVR)" value={result.mos_ovr} />
        </div>
      )}

      {error && (
        <p className="text-red-400 max-w-md text-center">Error: {error}</p>
      )}
    </main>
  );
}

function ScoreRow({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, Math.max(0, (value / 5) * 100));
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="font-mono">{value.toFixed(3)}</span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
