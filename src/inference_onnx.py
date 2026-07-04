import argparse
import onnxruntime as ort
from PIL import Image
from src import config
from src.pipeline import get_image_processor, image_to_pixel_values, load_id2label, postprocess


def load_session(providers=None) -> ort.InferenceSession:
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(config.ONNX_PATH), providers=providers)


def run(image_path: str, threshold: float = config.DEFAULT_THRESHOLD,
        session=None, processor=None, id2label=None):
    session = session or load_session()
    processor = processor or get_image_processor(local=True)
    id2label = id2label or load_id2label(local=True)

    image = Image.open(image_path).convert("RGB")
    pixel_values = image_to_pixel_values(image, processor)

    logits, pred_boxes = session.run(None, {"pixel_values": pixel_values})

    w, h = image.size
    return postprocess(
        logits, pred_boxes, processor,
        original_size=(h, w), id2label=id2label, threshold=threshold,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    args = ap.parse_args()

    dets = run(args.image, args.threshold)
    print(f"[ONNX] 偵測到 {len(dets)} 個目標：")
    for d in dets:
        print(f"  {d['label']:12s} conf={d['confidence']:.3f} box={d['box']}")
