"""Замер скорости инференса YOLOv11 на CPU Raspberry Pi 5: среднее время и FPS."""

import platform
import time
from pathlib import Path

from ultralytics import YOLO

INPUT = "images/input.jpg"
MODEL = "yolo11n.pt"
RUNS = 30
WARMUP = 3

Path("results").mkdir(exist_ok=True)
model = YOLO(MODEL)

for _ in range(WARMUP):
    model.predict(source=INPUT, verbose=False)

times = []
for _ in range(RUNS):
    start = time.perf_counter()
    model.predict(source=INPUT, verbose=False)
    times.append(time.perf_counter() - start)

avg_s = sum(times) / len(times)

report = f"""Замер скорости YOLOv11n на Raspberry Pi 5 (CPU)
=================================================
Машина:        {platform.node()} ({platform.machine()})
Python:        {platform.python_version()}
Модель:        YOLOv11n (CPU)
Картинка:      {INPUT}
Прогонов:      {RUNS} (+{WARMUP} на разогрев)

Среднее время на картинку: {avg_s * 1000:.1f} мс
Производительность:        {1 / avg_s:.2f} FPS
Минимум / максимум:        {min(times) * 1000:.1f} / {max(times) * 1000:.1f} мс
"""

print(report)
Path("results/benchmark.txt").write_text(report, encoding="utf-8")
