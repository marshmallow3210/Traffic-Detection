import argparse
import numpy as np
import torch
from PIL import Image
from transformers import YolosForObjectDetection
from src import config
from src.inference_onnx import load_session
from src.pipeline import get_image_processor, postprocess


def main(image_path: str, threshold: float = config.DEFAULT_THRESHOLD):
    processor = get_image_processor(local=False)
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

    # PyTorch 
    model = YolosForObjectDetection.from_pretrained(config.MODEL_NAME)
    model.eval()
    with torch.no_grad():
        pt_out = model(pixel_values=pixel_values)
    pt_logits, pt_boxes = pt_out.logits.numpy(), pt_out.pred_boxes.numpy()

    # ONNX 
    session = load_session()
    onnx_logits, onnx_boxes = session.run(
        None, {"pixel_values": pixel_values.numpy().astype(np.float32)}
    )

    # 原始輸出比對
    logits_diff = float(np.abs(pt_logits - onnx_logits).max())
    boxes_diff = float(np.abs(pt_boxes - onnx_boxes).max())
    print(f"logits     最大絕對誤差: {logits_diff:.6e}")
    print(f"pred_boxes 最大絕對誤差: {boxes_diff:.6e}")

    tol = 1e-3
    ok = logits_diff < tol and boxes_diff < tol
    print(f"數值一致性 (tol={tol}): {'✓ 通過' if ok else '✗ 超出容忍範圍'}")

    # 最終偵測結果比對 
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    w, h = image.size
    pt_dets = postprocess(pt_logits, pt_boxes, processor, (h, w), id2label, threshold)
    onnx_dets = postprocess(onnx_logits, onnx_boxes, processor, (h, w), id2label, threshold)
    print(f"\nPyTorch 偵測數: {len(pt_dets)}  |  ONNX 偵測數: {len(onnx_dets)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    args = ap.parse_args()
    main(args.image, args.threshold)
