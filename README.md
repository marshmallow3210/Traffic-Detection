# Traffic-Detection

Real-time object detection on live Taipei city traffic camera streams using YOLOv8s ONNX.

## Highlights

- Exports YOLOv8s from PyTorch `.pt` to ONNX, runs inference with ONNX Runtime
- Connects to Taipei City Government HLS live streams (`.m3u8`), runs inference every 3 frames
- FastAPI serves a live annotated video feed and a `/detect` JSON API
- Fully containerized with Docker

## Motivation

This project simulates the engineering workflow of deploying a CV model to an edge device (e.g. Nvidia Jetson):
PyTorch training → ONNX export → ONNX Runtime inference → API service → Docker

## Usage

```bash
cp .env.example .env  # add your stream URL
docker compose -f docker-compose.dev.yml run -p 8001:8000 dev
python -m src.stream
```

Open `http://localhost:8001` to view the live detection feed.