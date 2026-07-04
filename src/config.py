"""全域設定：模型、輸入尺寸、目標類別、輸出路徑。"""
from pathlib import Path

MODEL_NAME = "hustvl/yolos-tiny"

# 固定輸入尺寸 (H, W)，須可被 patch size 16 整除（512 / 16 = 32）。
# 為什麼固定 shape：
#   1. 邊緣部署 (Jetson / TensorRT) 通常為固定 shape 建置最佳化 engine。
#   2. 固定 shape 讓 YOLOS 的位置編碼插值在 trace 時變成常數，
#      可避開 ONNX 匯出動態插值的常見坑。
INPUT_SIZE = (512, 512)

# 只保留這些類別（路口的機車與行人）。
# 用「類別名稱」比對而非硬寫 id，避免不同 label set 的 id 對不上。
TARGET_CLASSES = {"person", "motorcycle"}

DEFAULT_THRESHOLD = 0.7

# 路徑
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
ONNX_PATH = MODELS_DIR / "yolos-tiny.onnx"
PROCESSOR_DIR = MODELS_DIR / "processor"
LABELS_PATH = MODELS_DIR / "labels.json"
