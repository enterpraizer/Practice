"""Сравнение скорости инференса YOLOv11n на CPU: 1 канал (gray) vs 3 канала (RGB).

1-канальная модель — настоящая: веса первого слоя суммируются по входным каналам
(для серого это даёт тот же результат, но честно с 1 каналом на входе).

Число потоков CPU ограничено (2 ядра), чтобы пиковая нагрузка не проваливала питание
платы. Абсолютные FPS из-за этого ниже 4-ядерных, но ВАЖНО СООТНОШЕНИЕ 1ch / 3ch —
оно от числа ядер не зависит.
"""

import copy
import time

import cv2
import torch
from ultralytics import YOLO

torch.set_num_threads(1)

IMG = "images/input.jpg"  # читаем цветную и переводим в серое прямо в коде (IMREAD_GRAYSCALE)
RUNS = 8
WARMUP = 2

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
            time.sleep(0.5)
        times = []
        for _ in range(RUNS):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
            time.sleep(0.5)
    avg = sum(times) / len(times)
    print(f"{channels} канал(а): {avg * 1000:7.1f} мс | {1 / avg:5.2f} FPS")
    return avg


print(f"YOLOv11n на CPU ({torch.get_num_threads()} ядра), ч/б картинка {IMG}:")
a3 = bench(3, model3)
time.sleep(2)
a1 = bench(1, model1)
print(f"\n1 канал быстрее в {a3 / a1:.3f}× ({(1 - a1 / a3) * 100:.1f}%)")
