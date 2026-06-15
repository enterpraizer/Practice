"""Распознавание объектов на ВИДЕО с помощью YOLO на ускорителе Hailo-8L (DeGirum) + замер FPS."""

import time
from pathlib import Path

import cv2
import degirum as dg
import degirum_tools

INPUT = "videos/input.mp4"
OUTPUT = "results/video_hailo.mp4"
MODEL = "yolov8s_coco--320x320_quant_hailort_hailo8l_1"

Path("results").mkdir(exist_ok=True)
token = Path("degirum_token.txt").read_text().strip()

model = dg.load_model(
    model_name=MODEL,
    inference_host_address="@local",
    zoo_url="degirum/hailo",
    token=token,
    device_type="HAILORT/HAILO8L",
)

# число кадров в видео — для расчёта FPS
cap = cv2.VideoCapture(INPUT)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# обрабатываем всё видео: инференс на каждом кадре + рамки + сохранение
start = time.perf_counter()
degirum_tools.annotate_video(model, INPUT, OUTPUT)
elapsed = time.perf_counter() - start

fps = total_frames / elapsed if elapsed > 0 else 0
print(f"Видео:     {INPUT}")
print(f"Результат: {OUTPUT}")
print(f"Кадров:    {total_frames}")
print(f"Время:     {elapsed:.1f} c")
print(f"Скорость:  {fps:.2f} FPS")
