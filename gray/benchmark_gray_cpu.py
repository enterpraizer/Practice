"""Сравнение скорости инференса YOLOv11n на CPU: 1 канал (gray) vs 3 канала (RGB).

1-канальная модель — настоящая: веса первого слоя суммируются по входным каналам
(для серого изображения это даёт тот же результат, но честно с 1 каналом на входе).
"""

import copy
import platform
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

torch.set_num_threads(4)

IMG = "images/input.jpg"  # читаем как ч/б (IMREAD_GRAYSCALE)
RUNS = 30
WARMUP = 5

Path("results").mkdir(exist_ok=True)

model3 = YOLO("yolo11n.pt").model.eval()

model1 = copy.deepcopy(model3)
first = model1.model[0].conv
first.weight = torch.nn.Parameter(first.weight.data.sum(dim=1, keepdim=True))
first.in_channels = 1


def tensor(channels):
    img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 640))
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t = torch.from_numpy(img).permute(2, 0, 1)
    else:
        t = torch.from_numpy(img).unsqueeze(0)
    return t.unsqueeze(0).float() / 255.0


def bench(channels, model):
    x = tensor(channels)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        times = []
        for _ in range(RUNS):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    return sum(times) / len(times)


a3 = bench(3, model3)
a1 = bench(1, model1)

report = f"""Скорость инференса YOLOv11n на CPU: 1 канал (gray) vs 3 канала (RGB)
=================================================
Машина:     {platform.node()} ({platform.machine()})
Ядер CPU:   {torch.get_num_threads()}
Картинка:   {IMG} (читается как ч/б)
Прогонов:   {RUNS} (+{WARMUP} на разогрев)

3 канала:  {a3 * 1000:6.1f} мс | {1 / a3:5.2f} FPS
1 канал:   {a1 * 1000:6.1f} мс | {1 / a1:5.2f} FPS

1 канал быстрее в {a3 / a1:.3f}x ({(1 - a1 / a3) * 100:.1f}%)
"""

print(report)
Path("results/benchmark_gray_cpu.txt").write_text(report, encoding="utf-8")
