"""Скорость 1-канальной (grey) YOLOv11n на CPU — model.predict, как rgb/benchmark.py.
Меряется ТОЛЬКО 1 канал. Сравнение — с RGB (3 канала) из rgb/benchmark.py.
1-канальная модель настоящая: веса первого слоя просуммированы по входным каналам.
"""

import time
from pathlib import Path

import cv2
import torch.nn as nn
from ultralytics import YOLO

COLOR = "images/input.jpg"
GRAY = "images/input_gray.jpg"
RUNS = 30
WARMUP = 3

Path("results").mkdir(exist_ok=True)
cv2.imwrite(GRAY, cv2.cvtColor(cv2.imread(COLOR), cv2.COLOR_BGR2GRAY))

m = YOLO("yolo11n.pt")
old = m.model.model[0].conv
new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
new.weight.data = old.weight.data.sum(dim=1, keepdim=True)
if old.bias is not None:
    new.bias.data = old.bias.data
m.model.model[0].conv = new
m.model.yaml["ch"] = 1
m.model.yaml["channels"] = 1

for _ in range(WARMUP):
    m.predict(GRAY, verbose=False)

times = []
for _ in range(RUNS):
    start = time.perf_counter()
    m.predict(GRAY, verbose=False)
    times.append(time.perf_counter() - start)

avg = sum(times) / len(times)

report = f"""Скорость YOLOv11n на CPU — model.predict: grey (1 канал) vs RGB (3 канала)
=================================================
Прогонов: {RUNS} (+{WARMUP} разогрев)

grey (1 канал):  {avg * 1000:6.1f} мс | {1 / avg:5.2f} FPS
RGB  (3 канала): 280.3 мс | 3.57 FPS   (из rgb/benchmark.py)
"""

print(report)
Path("results/benchmark_gray_cpu.txt").write_text(report, encoding="utf-8")
