"""1 канал (серый) vs 3 канала (цвет) — обе модели model.predict, в одинаковых условиях.

3 канала — на ЦВЕТНОЙ картинке (как прошлый эксперимент), 1 канал — на серой.
1-канальная модель настоящая: веса первого слоя просуммированы по входным каналам.
"""

import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO

torch.set_num_threads(1)

COLOR = "images/input.jpg"
GRAY = "images/input_gray.jpg"
RUNS = 30
WARMUP = 5

Path("results").mkdir(exist_ok=True)
cv2.imwrite(GRAY, cv2.cvtColor(cv2.imread(COLOR), cv2.COLOR_BGR2GRAY))

m3 = YOLO("yolo11n.pt")

m1 = YOLO("yolo11n.pt")
old = m1.model.model[0].conv
new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
new.weight.data = old.weight.data.sum(dim=1, keepdim=True)
if old.bias is not None:
    new.bias.data = old.bias.data
m1.model.model[0].conv = new
m1.model.yaml["ch"] = 1
m1.model.yaml["channels"] = 1


def bench(model, src):
    for _ in range(WARMUP):
        model.predict(src, verbose=False)
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        model.predict(src, verbose=False)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


a3 = bench(m3, COLOR)
a1 = bench(m1, GRAY)

report = f"""1 канал (серый) vs 3 канала (цвет) — CPU, model.predict, 1 ядро, одинаковые условия
=================================================
Прогонов: {RUNS} (+{WARMUP} разогрев)

3 канала (цвет):  {a3 * 1000:6.1f} мс | {1 / a3:5.2f} FPS
1 канал  (серый): {a1 * 1000:6.1f} мс | {1 / a1:5.2f} FPS
"""

print(report)
Path("results/benchmark_gray_cpu.txt").write_text(report, encoding="utf-8")
