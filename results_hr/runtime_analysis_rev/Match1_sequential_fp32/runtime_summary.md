# Runtime analysis — Match1

- Broadcast duration: **01:44:25.48** (156,637 frames at 25 fps, 1920×1080)
- Total pipeline compute: **11:30.21**
- Real-time factor: **0.110×** (compute / video duration; <1 means faster than real-time)
- Hardware: **CPU** 13th Gen Intel(R) Core(TM) i9-13900K (32 threads), **GPU** NVIDIA GeForce RTX 4090 (24564 MiB), NVIDIA GeForce RTX 4090 (24564 MiB), torch 2.10.0+cu128/CUDA 12.8
- Generated: 2026-11-19T12:43:10

## Per-substage compute time

| Stage | Substage | Time (s) | Share (%) | Throughput |
|---|---|---:|---:|---:|
| Shot segmentation | Per-frame histogram differences | 254.44 | 36.9 | 24.6× |
| Shot segmentation | Abrupt-cut detection | 0.01 | 0.0 | 578618.2× |
| Shot segmentation | Gradual-transition detection | 0.05 | 0.0 | 122697.1× |
| Shot segmentation | Shot assembly | 0.00 | 0.0 | 19140786.5× |
| Player detection (YOLOv8) | YOLOv8 model load | 1.86 | 0.3 | 3367.7× |
| Player detection (YOLOv8) | Video decoding of sampled frames | 32.09 | 4.6 | 195.3× |
| Player detection (YOLOv8) | YOLOv8 inference + post-processing | 218.42 | 31.6 | 28.7× |
| Play classification | Rolling smoothing + thresholding | 1.68 | 0.2 | 3721.0× |
| Scoreboard branch | IQR variability map | 16.96 | 2.5 | 369.3× |
| Scoreboard branch | Overlay mask construction | 0.03 | 0.0 | 225112.6× |
| Scoreboard branch | Scoreboard region search | 3.71 | 0.5 | 1689.4× |
| Scoreboard branch | OCR sampling + EasyOCR inference | 160.78 | 23.3 | 39.0× |
| Scoreboard branch | Multi-strategy parsing | 0.01 | 0.0 | 1062551.6× |
| Scoreboard branch | Robustness cascade | 0.00 | 0.0 | 1630799.3× |
| Highlight assembly | Play long-shot selection | 0.06 | 0.0 | 107918.8× |
| Highlight assembly | Goal-shot anchoring | 0.11 | 0.0 | 56957.3× |
| Evaluation | Load ground-truth annotations | 0.00 | 0.0 | 6776488.4× |
| Evaluation | Matching + scoring | 0.00 | 0.0 | 43229244.4× |
