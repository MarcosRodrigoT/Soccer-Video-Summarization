# Runtime analysis — Match1

- Broadcast duration: **01:44:25.48** (156,637 frames at 25 fps, 1920×1080)
- Total pipeline compute: **34:30.91**
- Real-time factor: **0.331×** (compute / video duration; <1 means faster than real-time)
- Hardware: **CPU** 13th Gen Intel(R) Core(TM) i9-13900K (32 threads), **GPU** NVIDIA GeForce RTX 4090 (24564 MiB), NVIDIA GeForce RTX 4090 (24564 MiB), torch 2.10.0+cu128/CUDA 12.8
- Generated: 2026-06-01T15:14:40

## Per-substage compute time

| Stage | Substage | Time (s) | Share (%) | Throughput |
|---|---|---:|---:|---:|
| Shot segmentation | Per-frame histogram differences | 275.42 | 13.3 | 22.7× |
| Shot segmentation | Abrupt-cut detection | 0.01 | 0.0 | 570021.0× |
| Shot segmentation | Gradual-transition detection | 0.05 | 0.0 | 120356.1× |
| Shot segmentation | Shot assembly | 0.00 | 0.0 | 17732156.0× |
| Player detection (YOLOv8) | YOLOv8 model load | 1.91 | 0.1 | 3277.5× |
| Player detection (YOLOv8) | Frame extraction + inference loop | 1616.66 | 78.1 | 3.9× |
| Play classification | Rolling smoothing + thresholding | 1.62 | 0.1 | 3863.4× |
| Scoreboard branch | IQR variability map | 17.75 | 0.9 | 353.0× |
| Scoreboard branch | Overlay mask construction | 0.03 | 0.0 | 227155.6× |
| Scoreboard branch | Scoreboard region search | 2.88 | 0.1 | 2178.5× |
| Scoreboard branch | OCR sampling + EasyOCR inference | 154.39 | 7.5 | 40.6× |
| Scoreboard branch | Multi-strategy parsing | 0.01 | 0.0 | 1147959.7× |
| Scoreboard branch | Robustness cascade | 0.00 | 0.0 | 1627312.4× |
| Highlight assembly | Play long-shot selection | 0.06 | 0.0 | 103844.1× |
| Highlight assembly | Goal-shot anchoring | 0.11 | 0.0 | 55655.1× |
| Evaluation | Load ground-truth annotations | 0.01 | 0.0 | 782927.1× |
| Evaluation | Matching + scoring | 0.00 | 0.0 | 43858096.3× |
