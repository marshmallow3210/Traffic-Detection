import argparse
import torch
from PIL import Image
from transformers import YolosForObjectDetection
from src import config
from src.pipeline import get_image_processor, postprocess


def run(image_path: str, threshold: float = config.DEFAULT_THRESHOLD):
    processor = get_image_processor(local=False)
    model = YolosForObjectDetection.from_pretrained(config.MODEL_NAME)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image.size
    return postprocess(
        outputs.logits, outputs.pred_boxes, processor,
        original_size=(h, w), id2label=id2label, threshold=threshold,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    args = ap.parse_args()

    dets = run(args.image, args.threshold)
    print(f"[PyTorch] 偵測到 {len(dets)} 個目標：")
    for d in dets:
        print(f"  {d['label']:12s} conf={d['confidence']:.3f} box={d['box']}")
