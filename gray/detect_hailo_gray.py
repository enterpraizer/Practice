"""1-канальная (grey) YOLOv11n на ускорителе Hailo-8L через DeGirum: рамки + замер FPS."""

import time
from pathlib import Path

import cv2
import degirum as dg

ZOO = "/home/nikita/yolo-pi/hailo_zoo_gray/yolov11n_gray/yolov11n_gray.json"
MODEL = "yolov11n_gray"
SRC = "images/input.jpg"
OUTPUT = "results/hailo_gray.jpg"
RUNS = 20
WARMUP = 3

Path("results").mkdir(exist_ok=True)

gray_img = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)[..., None]  # 1 канал (H, W, 1)

model = dg.load_model(
    model_name=MODEL,
    inference_host_address="@local",
    zoo_url=ZOO,
    device_type="HAILORT/HAILO8L",
)

result = model(gray_img)

overlay = result.image_overlay
try:
    overlay.save(OUTPUT)
except AttributeError:
    cv2.imwrite(OUTPUT, overlay)

print(f"Найдено объектов: {len(result.results)}")
for det in result.results:
    print(f"  - {det['label']}: {det['score']:.0%}")

for _ in range(WARMUP):
    model(gray_img)

times = []
for _ in range(RUNS):
    start = time.perf_counter()
    model(gray_img)
    times.append(time.perf_counter() - start)

avg = sum(times) / len(times)
print(f"\nСреднее время на картинку: {avg * 1000:.1f} мс")
print(f"Производительность:        {1 / avg:.2f} FPS")
