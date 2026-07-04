![CI](https://github.com/marshmallow3210/Traffic-Detection/actions/workflows/ci.yml/badge.svg)

# Traffic-Detection

Real-time object detection on live Taipei city traffic camera streams using YOLOv8s ONNX Runtime, served via FastAPI and containerized with Docker.

![detection demo](sample_images/demo.jpeg)

## Highlights

- Exports YOLOv8s from PyTorch `.pt` to ONNX format via `yolo export`, runs inference with ONNX Runtime
- Connects to Taipei City Government HLS live streams (`.m3u8`) via OpenCV, runs inference every 3 frames
- FastAPI serves a live annotated JPEG feed (`/feed`) and a JSON detection API (`/detect`)
- Fully containerized with Docker, stream URL injected via environment variable

## More Detections

![detection demo 2](sample_images/demo2.png)

## Benchmark: ONNX Runtime vs PyTorch (CPU)

![benchmark](sample_images/benchmark.png)

ONNX Runtime runs **~3x faster** than PyTorch on CPU (avg 78ms vs 246ms per frame, ranging 2.75x to 3.45x across 5 runs). It also uses about 40% less additional memory (154MB vs 267MB), which matters on memory-constrained edge devices. This gap is expected to widen further with TensorRT on edge hardware.

Timing covers the full per-frame pipeline (preprocess + inference + postprocess), with 10 warmup frames excluded. Both frameworks run on identical input frames captured from the live stream. Measured on MacBook Air (M5, 32GB) inside a Docker container (linux/arm64), CPU execution only.

## Pipeline

```
HLS Stream → OpenCV → Preprocess (640×640) → ONNX Runtime → Postprocess + NMS → FastAPI → Browser
```

## Motivation

Simulates the engineering workflow of deploying a CV model to an edge device (e.g. Nvidia Jetson):

```
PyTorch training → ONNX export → ONNX Runtime inference → API service → Docker
```

The same ONNX model can switch execution providers (CPU, CUDA, TensorRT) with a single line change, no model modification needed.

## Getting a Stream URL

This project uses Taipei City's public traffic cameras.

1. Open https://its.taipei.gov.tw and pick an intersection camera
2. Open browser DevTools (F12) → Network tab, filter by `m3u8`
3. Play the stream, copy the `.m3u8` request URL
4. Paste it into `.env` as `HLS_URL=...`

Note: stream URLs contain a token and expire after some time.
If the app logs "Failed to open stream", re-extract a fresh URL.

## Setup (Development)

Step 1. On your host machine, set up the stream URL.

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with a real HLS stream URL.

> **Note:** `.env` is loaded when the container starts. If you edit it after
> starting the container, exit and re-run it for changes to take effect.

Step 2. On your host machine, start the dev container.

```bash
docker compose -f docker-compose.dev.yml run -p 8001:8000 dev
```

This drops you into a shell inside the container.

> **Note:** If port 8001 is already in use, pick another one, e.g. `-p 8003:8000`.

Step 3. Inside the container, export the ONNX model. This only needs to run once.

```bash
mkdir -p models
yolo export model=yolov8s.pt format=onnx opset=17
mv yolov8s.onnx models/
```

Step 4. Inside the container, load the stream URL and start the service.

```bash
python -m src.stream
```

Step 5. On your host machine, open a browser.

```
http://localhost:8001
```

Run the benchmark (inside the container, after step 3):

```bash
python -m src.benchmark --frames 50
```

## Setup (Production)

This builds a self contained image with the model and code baked in.

Step 1. Make sure `models/yolov8s.onnx` exists locally (produced in the Development steps above).

Step 2. On your host machine, build and run the image.

```bash
docker build -t traffic-detection .
docker run -p 8002:8000 --env HLS_URL="your_stream_url" traffic-detection
```

Step 3. On your host machine, open a browser.

```
http://localhost:8002
```

Only the stream URL is passed at runtime. No further setup is needed.

## Known Limitations

- Motorcycle and bicycle detection is unreliable at overhead angles due to domain shift from COCO training data. Riders are often classified as person when viewed from above. Fine tuning on local traffic footage would improve this.
- Inference runs on CPU only. Switching to `TensorrtExecutionProvider` on Jetson would reduce latency from about 78ms to under 10ms.
- Stream URL may expire and require re extraction from browser DevTools.
- NMS occasionally leaves overlapping boxes on the same object (e.g. two bus boxes on one vehicle). Tuning `NMS_IOU` in `src/stream.py` may help.