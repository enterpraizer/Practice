"""Замер скорости инференса YOLO на ускорителе Hailo-8L (DeGirum): среднее время и FPS."""

import platform
import time
from pathlib import Path

import degirum as dg

INPUT = "images/input.jpg"
MODEL = "yolov8s_coco--320x320_quant_hailort_hailo8l_1"
RUNS = 50
WARMUP = 5  # прогоны на разогрев, в статистику не идут

Path("results").mkdir(exist_ok=True)
token = Path("degirum_token.txt").read_text().strip()

model = dg.load_model(
    model_name=MODEL,
    inference_host_address="@local",
    zoo_url="degirum/hailo",
    token=token,
    device_type="HAILORT/HAILO8L",
)

for _ in range(WARMUP):
    model(INPUT)

times = []
for _ in range(RUNS):
    start = time.perf_counter()
    model(INPUT)
    times.append(time.perf_counter() - start)

avg_s = sum(times) / len(times)

report = f"""Замер скорости YOLOv8s на Hailo-8L (ускоритель)
=================================================
Машина:        {platform.node()} ({platform.machine()})
Модель:        YOLOv8s COCO 320x320 (Hailo-8L)
Картинка:      {INPUT}
Прогонов:      {RUNS} (+{WARMUP} на разогрев)

Среднее время на картинку: {avg_s * 1000:.1f} мс
Производительность:        {1 / avg_s:.2f} FPS
Минимум / максимум:        {min(times) * 1000:.1f} / {max(times) * 1000:.1f} мс
"""

print(report)
Path("results/benchmark_hailo.txt").write_text(report, encoding="utf-8")
