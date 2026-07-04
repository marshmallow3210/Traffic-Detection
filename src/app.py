import io
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from src import config
from src.inference_onnx import load_session
from src.pipeline import get_image_processor, image_to_pixel_values, load_id2label, postprocess


_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["session"] = load_session()
    _state["processor"] = get_image_processor(local=True)
    _state["id2label"] = load_id2label(local=True)
    yield
    _state.clear()


app = FastAPI(title="Traffic Detection ONNX", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": config.MODEL_NAME,
        "input_size": list(config.INPUT_SIZE),
        "target_classes": sorted(config.TARGET_CLASSES),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), threshold: float = config.DEFAULT_THRESHOLD):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    pixel_values = image_to_pixel_values(image, _state["processor"])

    t0 = time.perf_counter()
    logits, pred_boxes = _state["session"].run(None, {"pixel_values": pixel_values})
    infer_ms = round((time.perf_counter() - t0) * 1000, 2)

    w, h = image.size
    detections = postprocess(
        logits, pred_boxes, _state["processor"], (h, w),
        _state["id2label"], threshold,
    )
    return {
        "detections": detections,
        "count": len(detections),
        "inference_ms": infer_ms,
    }
