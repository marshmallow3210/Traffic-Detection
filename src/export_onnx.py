import json
import onnx
import torch
from transformers import YolosForObjectDetection
from src import config
from src.pipeline import get_image_processor


class YolosOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 載入模型 {config.MODEL_NAME} ...")
    model = YolosForObjectDetection.from_pretrained(config.MODEL_NAME)
    model.eval()

    wrapper = YolosOnnxWrapper(model)
    h, w = config.INPUT_SIZE
    dummy = torch.randn(1, 3, h, w)

    print(f"[2/4] 匯出 ONNX -> {config.ONNX_PATH} ...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(config.ONNX_PATH),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        # 只讓 batch 維度動態；空間維度固定，符合邊緣部署且避開動態插值坑。
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print("[3/4] 用 onnx.checker 驗證模型結構 ...")
    onnx_model = onnx.load(str(config.ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("      ✓ ONNX 結構驗證通過")

    print("[4/4] 儲存前處理器與標籤（供離線推論）...")
    processor = get_image_processor(local=False)
    processor.save_pretrained(str(config.PROCESSOR_DIR))
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    print(f"      ✓ 前處理器 -> {config.PROCESSOR_DIR}")
    print(f"      ✓ 標籤     -> {config.LABELS_PATH}")
    print("\n完成。")


if __name__ == "__main__":
    main()
