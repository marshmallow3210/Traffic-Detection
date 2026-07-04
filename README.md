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

ONNX Runtime runs **2.59x faster** than PyTorch on CPU (avg 99ms vs 257ms per frame), with no loss in detection accuracy. This gap is expected to widen further on edge hardware with TensorRT.

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

## Setup (Development)

Step 1. On your host machine, set up the stream URL.

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with a real HLS stream URL.

Step 2. On your host machine, start the dev container.

```bash
docker compose -f docker-compose.dev.yml run -p 8001:8000 dev
```

This drops you into a shell inside the container.

Step 3. Inside the container, export the ONNX models. This only needs to run once.

```bash
yolo export model=yolov8s.pt format=onnx opset=17
mv yolov8s.onnx models/
```

Step 4. Inside the container, load the stream URL and start the service.

```bash
export HLS_URL=$(grep HLS_URL .env | cut -d= -f2)
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
- Inference runs on CPU only. Switching to `TensorrtExecutionProvider` on Jetson would reduce latency from about 90ms to under 10ms.
- Stream URL may expire and require re extraction from browser DevTools.
- NMS occasionally leaves overlapping boxes on the same object (e.g. two bus boxes on one vehicle). Tuning `NMS_IOU` in `src/stream.py` may help.