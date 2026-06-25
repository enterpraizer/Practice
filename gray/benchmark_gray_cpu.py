"""Скорость 1-канальной (серой) YOLOv11n на картинке — model.predict, как в прошлом цветном замере.

Меряется ТОЛЬКО 1-канальная модель. Сравнение — с прошлым цветным экспериментом (3 канала, 2.7 FPS).
1-канальная модель настоящая: веса первого слоя просуммированы по входным каналам.
"""

import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO

torch.set_num_threads(4)

COLOR = "images/input.jpg"
GRAY = "images/input_gray.jpg"
RUNS = 30
WARMUP = 5

Path("results").mkdir(exist_ok=True)
cv2.imwrite(GRAY, cv2.cvtColor(cv2.imread(COLOR), cv2.COLOR_BGR2GRAY))

# создаём 1-канальную модель
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

report = f"""Скорость 1-канальной (серой) YOLOv11n на CPU — model.predict
=================================================
Картинка: {GRAY}, прогонов: {RUNS} (+{WARMUP} разогрев)

1 канал (серый): {avg * 1000:.1f} мс | {1 / avg:.2f} FPS

Прошлый эксперимент (3 канала, цвет, тот же model.predict): 2.70 FPS / 371 мс
"""

print(report)
Path("results/benchmark_gray_cpu.txt").write_text(report, encoding="utf-8")
