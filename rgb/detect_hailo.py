"""Наша YOLOv11n (.hef) на ускорителе Hailo-8L через DeGirum (локальная модель): рамки + замер FPS."""

import time
from pathlib import Path

import degirum as dg

ZOO = "/home/nikita/yolo-pi/hailo_zoo/yolov11n/yolov11n.json"
MODEL = "yolov11n"
INPUT = "images/input.jpg"
OUTPUT = "results/hailo.jpg"
RUNS = 20
WARMUP = 3

Path("results").mkdir(exist_ok=True)

model = dg.load_model(
    model_name=MODEL,
    inference_host_address="@local",
    zoo_url=ZOO,
    device_type="HAILORT/HAILO8L",
)

result = model(INPUT)

overlay = result.image_overlay
try:
    overlay.save(OUTPUT)
except AttributeError:
    import cv2
    cv2.imwrite(OUTPUT, overlay)

print(f"Обработано: {INPUT}")
print(f"Результат:  {OUTPUT}")
print(f"Найдено объектов: {len(result.results)}")
for det in result.results:
    print(f"  - {det['label']}: {det['score']:.0%}")

for _ in range(WARMUP):
    model(INPUT)

times = []
for _ in range(RUNS):
    start = time.perf_counter()
    model(INPUT)
    times.append(time.perf_counter() - start)

avg_s = sum(times) / len(times)
print(f"\nСреднее время на картинку: {avg_s * 1000:.1f} мс")
print(f"Производительность:        {1 / avg_s:.2f} FPS")
