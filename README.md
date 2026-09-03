# COFFEEBEAN

Two separate, offline desktop demonstrations:

1. A deterministic comparison of FXLMS, conventional FXLMM, and modified
   FXLMM under impulsive noise.
2. A pretrained DeepFilterNet3 speech-enhancement benchmark.

This repository does **not** drive a speaker or implement a live physical ANC
loop. Do not test acoustic feedback ANC without a calibrated reference mic,
error mic, secondary-path model, output limiter, and hearing-safety controls.

## Setup

The core paper simulation and metrics use Python 3.12. DeepFilterNet is an
optional, substantially larger install.

```bash
uv python install 3.12
uv sync
uv run pytest -q

# Add pretrained enhancement support only when needed.
uv sync --extra enhance

# Add microphone capture support.
uv sync --extra enhance --extra live

# Build the small native RNNoise v0.2 library (no Python model dependency).
./scripts/build-rnnoise-v0.2.sh
```

Generated audio and reports belong under `runs/`, which Git ignores. Local
recordings belong under `samples/local/` or `samples/real-noise/` and are also
ignored.

## 1. Reproduce the paper experiment

```bash
uv run coffeebean simulate-paper \
  --seed 1969587 \
  --output runs/paper
```

The command writes:

- `metrics.json` with residual, impulse-window, recovery, and coefficient norms
- `residuals.csv` with every signal trace
- `comparison.png` for visual comparison
- the reference and residual signals as floating-point WAV files

The paper does not publish its primary/secondary path coefficients, exact
impulse width, or complete online threshold estimator. The implementation uses
small documented lab FIR paths, 100 ms bursts at 9.5 s and 14.8 s, and rolling
95th/97.5th/99th percentiles. Its modified score is the continuous triangle
described by the paper's prose and Figure 1; this avoids the typesetting
inconsistency in Equation 1. The result is a qualitative, not pixel-identical,
reproduction.

## 2. Test speech enhancement on this Mac

Record 10-30 seconds of clean speech using QuickTime Player's **File > New
Audio Recording**. Convert it to mono 48 kHz WAV if needed:

```bash
mkdir -p samples/local runs/local
ffmpeg -i clean.m4a -ar 48000 -ac 1 samples/local/clean.wav
```

Create deterministic continuous-noise cases at -5, 0, and +5 dB, plus an
impulsive stress case:

```bash
for snr in -5 0 5; do
  uv run coffeebean mix \
    --clean samples/local/clean.wav \
    --profile continuous \
    --snr "$snr" \
    --output "runs/local/noisy-${snr}db.wav"
done

uv run coffeebean mix \
  --clean samples/local/clean.wav \
  --profile impulsive \
  --snr 0 \
  --output runs/local/noisy-impulsive.wav
```

Enhance one case and calculate metrics:

```bash
uv run coffeebean enhance \
  --input runs/local/noisy-0db.wav \
  --output runs/local/enhanced.wav \
  --model DeepFilterNet3

uv run coffeebean evaluate \
  --clean samples/local/clean.wav \
  --noisy runs/local/noisy-0db.wav \
  --enhanced runs/local/enhanced.wav \
  --output runs/local/metrics.json

afplay runs/local/noisy-0db.wav
afplay runs/local/enhanced.wav
```

## 3. Record-and-enhance microphone prototype

List microphone inputs, then record an exact-duration sample. macOS will ask
for microphone permission on the first capture:

```bash
uv run coffeebean devices

uv run coffeebean live \
  --device 0 \
  --duration 10 \
  --output runs/live

afplay runs/live/noisy.wav
afplay runs/live/enhanced.wav
```

`runs/live/report.json` records capture latency, input overflow blocks,
clipping, and enhancement runtime. This command records first and enhances
immediately afterward; it is deliberately reported as non-streaming and does
not play anti-noise through a speaker.

For bounded real-time processing during capture, use the separate `stream`
command. It keeps one chunk of look-ahead, writes both streams incrementally,
and reports every missed processing deadline:

```bash
uv run coffeebean stream \
  --device 0 \
  --duration 10 \
  --chunk-ms 40 \
  --context-ms 500 \
  --output runs/stream

afplay runs/stream/noisy.wav
afplay runs/stream/enhanced.wav
```

This Python prototype does not monitor through speakers. Use the native edge
runtime for a production communications stream with a virtual microphone.

RNNoise v0.2 little can be selected without replacing DeepFilterNet3:

```bash
./scripts/build-rnnoise-v0.2.sh

uv run coffeebean enhance \
  --input runs/mac-reference/gunfire-noisy.wav \
  --output runs/mac-reference/gunfire-rnnoise-v0.2.wav \
  --model RNNoise
```

The sparse/int8-weight native library measured 876 KiB with 32,688 bytes of
per-stream state. On the 16.1-second 0 dB gunfire case it achieved RTF 0.033,
8.49 dB SI-SDR improvement, and 0.027 STOI improvement on this Mac.
DeepFilterNet3 achieved 14.66 dB SI-SDR improvement, 0.070 STOI improvement,
and RTF 0.018 on the same case. RNNoise remains valuable because
its native deployment is much smaller; these single-recording figures are a
demo, not a general quality claim.

The size-first edge target and its timing gate are documented in
[edge/README.md](edge/README.md). Raspberry Pi Zero 2 W is provisional until it
passes the on-device 60-second test; use Raspberry Pi 4 2 GB if it does not.

## 4. Hackathon dashboard

Launch the judge-facing desktop dashboard:

```bash
uv run coffeebean-demo
```

Choose the microphone and duration, then click **Start AI stream**. The window
shows noisy/enhanced waveforms and spectrograms, streaming deadline and latency
metrics, and controlled A/B playback after capture. Outputs remain under
`runs/demo/`; speaker monitoring stays disabled during recording.

`enhance` writes a neighboring benchmark JSON with wall-clock time and
real-time factor. For this desktop milestone, a 60-second file should have a
real-time factor below 1.0. At 0 and +5 dB, use positive median SI-SDR
improvement and non-regressing STOI as the quality gate. Treat -5 dB and
impulsive results as reported stress tests.

To mix a field recording instead of generated noise, pass `--noise`; the mixer
resamples it to mono 48 kHz, repeats short clips to the clean-speech duration,
and records the source path and crest factor in the neighboring JSON file:

```bash
uv run coffeebean mix \
  --clean samples/local/clean.wav \
  --noise samples/real-noise/wav/gunfire.wav \
  --snr 0 \
  --output runs/real-noise/gunfire-noisy.wav
```

The exact recordings and licenses used for the real-noise benchmark are listed
in [REAL_NOISE_SOURCES.md](REAL_NOISE_SOURCES.md). Real gunfire remains an
offline recording-only stress test; this prototype does not detect, localize,
or physically cancel gunfire.

PESQ is intentionally omitted because it is a withdrawn legacy standard.
POLQA is deferred because it requires licensed tooling. Neither affects the
audio or evaluation interfaces.
