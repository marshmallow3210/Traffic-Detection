import argparse
import gc
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
from ultralytics import YOLO
import onnxruntime as ort

from src.stream import preprocess as stream_preprocess, postprocess as stream_postprocess

ONNX_PATH = "/app/models/yolov8s.onnx"
PT_PATH = "yolov8s.pt"
HLS_URL = os.environ.get("HLS_URL", "")
WARMUP = 10


def grab_frames(n):
    print(f"從串流抓取 {n} 幀...")
    cap = cv2.VideoCapture(HLS_URL)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"取得 {len(frames)} 幀")
    return frames


def benchmark_onnx(frames):
    print("ONNX Runtime 推論中...")
    gc.collect()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024 / 1024
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for frame in frames[:WARMUP]:
        blob, scale, px, py = stream_preprocess(frame)
        session.run(None, {input_name: blob})

    times = []
    for frame in frames:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        blob, scale, px, py = stream_preprocess(frame)
        output = session.run(None, {input_name: blob})
        stream_postprocess(output, scale, px, py, w, h)
        times.append((time.perf_counter() - t0) * 1000)

    mem_after = proc.memory_info().rss / 1024 / 1024
    del session
    gc.collect()
    return times, mem_after - mem_before


def benchmark_pytorch(frames):
    print("PyTorch 推論中...")
    gc.collect()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024 / 1024
    model = YOLO(PT_PATH)

    for frame in frames[:WARMUP]:
        model(frame, verbose=False)

    times = []
    for frame in frames:
        t0 = time.perf_counter()
        model(frame, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    mem_after = proc.memory_info().rss / 1024 / 1024
    del model
    gc.collect()
    return times, mem_after - mem_before


def plot(onnx_times, pt_times, onnx_mem, pt_mem, n):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PyTorch vs ONNX Runtime (YOLOv8s, CPU)", fontsize=14, fontweight="bold")
    ax1.plot(range(1, n + 1), pt_times, label=f"PyTorch  avg={np.mean(pt_times):.1f}ms", color="#EE6C4D", linewidth=1.5)
    ax1.plot(range(1, n + 1), onnx_times, label=f"ONNX Runtime  avg={np.mean(onnx_times):.1f}ms", color="#3D8EAE", linewidth=1.5)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Inference Time (ms)")
    ax1.set_title("End-to-End Time per Frame (pre + infer + post)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    bars = ax2.bar(["PyTorch", "ONNX Runtime"], [pt_mem, onnx_mem], color=["#EE6C4D", "#3D8EAE"], width=0.4)
    ax2.set_ylabel("Memory Usage (MB)")
    ax2.set_title("Additional Memory Usage")
    for bar, val in zip(bars, [pt_mem, onnx_mem]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f} MB", ha="center", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    speedup = np.mean(pt_times) / np.mean(onnx_times)
    fig.text(0.5, 0.01, f"ONNX Runtime is {speedup:.2f}x faster than PyTorch on CPU",
             ha="center", fontsize=12, color="#2D6A4F", fontweight="bold")
    out_path = "/app/sample_images/benchmark.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"圖表已存到 {out_path}")


def main(n):
    frames = grab_frames(n)
    if not frames:
        print("無法取得幀，請確認 HLS_URL 環境變數")
        return
    onnx_times, onnx_mem = benchmark_onnx(frames)
    pt_times, pt_mem = benchmark_pytorch(frames)
    print(f"\nPyTorch      平均時間: {np.mean(pt_times):.1f}ms")
    print(f"ONNX Runtime 平均時間: {np.mean(onnx_times):.1f}ms")
    print(f"加速比: {np.mean(pt_times)/np.mean(onnx_times):.2f}x")
    print(f"\nPyTorch      記憶體增量: {pt_mem:.1f}MB")
    print(f"ONNX Runtime 記憶體增量: {onnx_mem:.1f}MB")
    plot(onnx_times, pt_times, onnx_mem, pt_mem, len(frames))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    args = ap.parse_args()
    main(args.frames)