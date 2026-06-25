"""Сравнение скорости инференса YOLOv11n на CPU: 3 канала (RGB) vs 1 канал (grayscale).

1-канальная модель получается из обычной суммированием весов первого слоя по входным каналам
(для серого изображения это даёт тот же результат, но честно с 1 каналом на входе).
"""

import copy
import time

import torch
from ultralytics import YOLO

RUNS = 30
WARMUP = 3

model3 = YOLO("yolo11n.pt").model.eval()

model1 = copy.deepcopy(model3)
first = model1.model[0].conv
w = first.weight.data
first.weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
first.in_channels = 1


def bench(model, channels):
    x = torch.randn(1, channels, 640, 640)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        times = []
        for _ in range(RUNS):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    print(f"{channels} канал(а): {avg * 1000:6.1f} мс | {1 / avg:5.2f} FPS")
    return avg


print("Инференс YOLOv11n на CPU (640x640):")
a3 = bench(model3, 3)
a1 = bench(model1, 1)
print(f"\nРазница: 1 канал быстрее в {a3 / a1:.2f}× ({(1 - a1 / a3) * 100:.1f}%)")
