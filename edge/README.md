# RNNoise v0.2 edge target

Use the official RNNoise v0.2 **little** model on a 64-bit Linux SBC. Its main
recurrent weights are sparse int8 values, debug float duplicates are excluded,
and the same C library builds on macOS and ARM Linux.

Measured on the development Mac:

- library: 901,120 bytes of mapped image (876 KiB file)
- per-stream state: 32,688 bytes
- audio: mono, 48 kHz, 480 samples per frame
- compensated algorithmic latency: 960 samples (20 ms)
- 16.1-second gunfire case: RTF 0.033

The size-first target is Raspberry Pi Zero 2 W. Treat it as provisional until
the same 60-second test reports RTF below 0.8 on that board. Raspberry Pi 4
2 GB is the fallback for a reliable hackathon demonstration.

On either macOS or the board:

```bash
# Raspberry Pi OS only:
sudo apt install build-essential curl

./scripts/build-rnnoise-v0.2.sh
uv sync --extra live

uv run coffeebean enhance \
  --input runs/real-noise/gunfire-noisy.wav \
  --output runs/real-noise/gunfire-rnnoise-v0.2.wav \
  --model RNNoise

uv run coffeebean stream \
  --device 0 \
  --duration 60 \
  --chunk-ms 20 \
  --context-ms 40 \
  --model RNNoise \
  --output runs/edge-smoke
```

Do not claim ESP32 support from these size figures alone. The model fits many
MCU flash/RAM budgets, but sustained 48 kHz inference still needs a measured
real-time port and optimized kernels.
