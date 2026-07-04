import json
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from transformers import YolosImageProcessor

from src import config


def get_image_processor(local: bool = False) -> YolosImageProcessor:
    src_path = str(config.PROCESSOR_DIR) if local else config.MODEL_NAME
    h, w = config.INPUT_SIZE
    return YolosImageProcessor.from_pretrained(
        src_path,
        size={"height": h, "width": w},
    )


def load_id2label(local: bool = False) -> dict:
    if local and config.LABELS_PATH.exists():
        with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    from transformers import YolosForObjectDetection
    model = YolosForObjectDetection.from_pretrained(config.MODEL_NAME)
    return {int(k): v for k, v in model.config.id2label.items()}


def postprocess(logits, pred_boxes, processor, original_size, id2label,
                threshold: float = config.DEFAULT_THRESHOLD):
    outputs = SimpleNamespace(
        logits=torch.as_tensor(logits),
        pred_boxes=torch.as_tensor(pred_boxes),
    )
    target_sizes = torch.tensor([list(original_size)])  # [[H, W]]
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        name = id2label.get(int(label_id), str(int(label_id)))
        if name not in config.TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = [round(float(v), 2) for v in box.tolist()]
        detections.append({
            "label": name,
            "confidence": round(float(score), 4),
            "box": [x1, y1, x2, y2],
        })
    return detections


def image_to_pixel_values(image: Image.Image, processor) -> np.ndarray:
    inputs = processor(images=image.convert("RGB"), return_tensors="np")
    return inputs["pixel_values"].astype(np.float32)
