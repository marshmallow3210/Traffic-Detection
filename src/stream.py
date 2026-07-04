import io
import os
import os
import time
import threading
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime, timezone, timedelta
TAIPEI = timezone(timedelta(hours=8))
from PIL import Image
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

HLS_URL = os.environ.get("HLS_URL", "")  # 設定在 .env 或環境變數
ONNX_PATH = "/app/models/yolov8s.onnx"
INPUT_SIZE = (640, 640)
THRESHOLD = 0.45
NMS_IOU = 0.45
SKIP = 3

COCO_NAMES = {
    0:"person", 1:"bicycle", 2:"car", 3:"motorcycle",
    5:"bus", 7:"truck"
}
TARGET = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
COLORS = {
    "person":     (0, 255, 0),
    "bicycle":    (0, 255, 255),
    "motorcycle": (0, 165, 255),
    "car":        (255, 255, 0),
    "truck":      (0, 0, 255),
    "bus":        (255, 0, 255),
}

app = FastAPI()
state = {"frame": b"", "stats": "等待中..."}

def preprocess(frame_bgr):
    h, w = frame_bgr.shape[:2]
    ih, iw = INPUT_SIZE
    scale = min(iw / w, ih / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (nw, nh))
    canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (iw - nw) // 2, (ih - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob[np.newaxis], scale, pad_x, pad_y

def postprocess(output, scale, pad_x, pad_y, orig_w, orig_h):
    preds = output[0].squeeze()
    boxes_xywh = preds[:4].T
    scores = preds[4:].T
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    mask = confidences >= THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    boxes_xyxy, confs, labels = [], [], []
    for (cx, cy, bw, bh), conf, cid in zip(boxes_xywh, confidences, class_ids):
        name = COCO_NAMES.get(int(cid))
        if name not in TARGET:
            continue
        x1 = max(0, int((cx - bw/2 - pad_x) / scale))
        y1 = max(0, int((cy - bh/2 - pad_y) / scale))
        x2 = min(orig_w, int((cx + bw/2 - pad_x) / scale))
        y2 = min(orig_h, int((cy + bh/2 - pad_y) / scale))
        boxes_xyxy.append([x1, y1, x2, y2])
        confs.append(float(conf))
        labels.append(name)

    if not boxes_xyxy:
        return []
    indices = cv2.dnn.NMSBoxes(
        [[x,y,x2-x,y2-y] for x,y,x2,y2 in boxes_xyxy],
        confs, THRESHOLD, NMS_IOU
    )
    return [(labels[i], confs[i], boxes_xyxy[i]) for i in indices.flatten()]

def draw_detections(frame_bgr, dets, ms):
    out = frame_bgr.copy()
    for label, conf, (x1, y1, x2, y2) in dets:
        color = COLORS.get(label, (255,255,255))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(out, text, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    ts = datetime.now(TAIPEI).strftime("%H:%M:%S")
    for col, thick in [((0,0,0),3), ((255,255,255),1)]:
        cv2.putText(out, f"{ts}  {ms:.0f}ms", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, thick)
    return out

def inference_loop():
    print("載入 YOLOv8s ONNX 模型...")
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("模型載入完成！")

    while True:
        try:
            print(f"連線到：{HLS_URL}")
            cap = cv2.VideoCapture(HLS_URL)
            if not cap.isOpened():
                print("無法開啟串流，10 秒後重試...")
                time.sleep(10)
                continue

            print("串流開始！")
            frame_count = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("串流中斷，重新連線...")
                    break

                frame_count += 1
                if frame_count % SKIP != 0:
                    continue

                h, w = frame_bgr.shape[:2]
                blob, scale, pad_x, pad_y = preprocess(frame_bgr)

                t0 = time.perf_counter()
                output = session.run(None, {input_name: blob})
                ms = (time.perf_counter() - t0) * 1000

                dets = postprocess(output, scale, pad_x, pad_y, w, h)
                counts = {}
                for label, _, _ in dets:
                    counts[label] = counts.get(label, 0) + 1

                summary = "  ".join(f"{k}:{v}" for k, v in counts.items())
                ts = datetime.now(TAIPEI).strftime("%H:%M:%S")
                print(f"[{ts}] {ms:.0f}ms | {summary or '無目標'}")

                annotated = draw_detections(frame_bgr, dets, ms)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                state["frame"] = buf.tobytes()
                state["stats"] = summary or "無目標"

            cap.release()
        except Exception as e:
            print(f"錯誤：{e}，10 秒後重試...")
            time.sleep(10)

@app.get("/feed")
def feed():
    if not state["frame"]:
        return Response("等待中...", media_type="text/plain")
    return Response(state["frame"], media_type="image/jpeg")

@app.get("/stats")
def stats():
    return {"stats": state["stats"]}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
      <title>Traffic Detection Live</title>
      <style>
        body{background:#111;color:#eee;font-family:sans-serif;text-align:center;padding:20px;margin:0}
        img{max-width:100%;border-radius:8px}
        .legend{margin:10px 0;font-size:14px;color:#aaa}
        .dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:4px;vertical-align:middle}
        #stats{margin-top:8px;font-size:18px;min-height:26px;color:#4fc;font-weight:bold}
      </style>
    </head>
    <body>
      <h2>🚦 133-松高松智路口 即時偵測 (YOLOv8s ONNX)</h2>
      <img id="feed" src="/feed"/>
      <div class="legend">
        <span class="dot" style="background:lime"></span>person　
        <span class="dot" style="background:cyan"></span>bicycle　
        <span class="dot" style="background:orange"></span>motorcycle　
        <span class="dot" style="background:yellow"></span>car　
        <span class="dot" style="background:blue"></span>truck　
        <span class="dot" style="background:magenta"></span>bus
      </div>
      <div id="stats">偵測中...</div>
      <script>
        function refresh(){
          const img=new Image();
          img.onload=()=>{document.getElementById('feed').src=img.src};
          img.src='/feed?t='+Date.now();
        }
        setInterval(refresh, 800);
        setInterval(()=>{
          fetch('/stats').then(r=>r.json()).then(d=>{
            document.getElementById('stats').textContent=d.stats;
          });
        }, 800);
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
