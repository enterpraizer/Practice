"""Сравнение скорости YOLOv11n на CPU полным конвейером (как старый цветной замер):
1 канал (серый, сейчас) vs 3 канала (цвет).

Меряем загрузка + предобработка + инференс + NMS — тот же объём, что у model.predict,
чтобы число серого было сравнимо со старым цветным результатом (2.7 FPS).
1-канальная модель — настоящая (веса первого слоя просуммированы по каналам).
"""

import copy
import platform
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.nms import non_max_suppression
except (ImportError, AttributeError):
    from ultralytics.utils.ops import non_max_suppression

torch.set_num_threads(4)

IMG = "images/input.jpg"
RUNS = 30
WARMUP = 5

Path("results").mkdir(exist_ok=True)

model3 = YOLO("yolo11n.pt").model.eval()

model1 = copy.deepcopy(model3)
first = model1.model[0].conv
first.weight = torch.nn.Parameter(first.weight.data.sum(dim=1, keepdim=True))
first.in_channels = 1


def prep(channels):
    img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 640))
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t = torch.from_numpy(img).permute(2, 0, 1)
    else:
        t = torch.from_numpy(img).unsqueeze(0)
    return t.unsqueeze(0).float() / 255.0


def bench(channels, model):
    with torch.no_grad():
        for _ in range(WARMUP):
            non_max_suppression(model(prep(channels)))
        times = []
        for _ in range(RUNS):
            start = time.perf_counter()
            x = prep(channels)
            preds = model(x)
            non_max_suppression(preds)
            times.append(time.perf_counter() - start)
    return sum(times) / len(times)


a3 = bench(3, model3)
a1 = bench(1, model1)

report = f"""Скорость YOLOv11n на CPU (полный конвейер: загрузка+препроцесс+инференс+NMS)
=================================================
Машина:   {platform.node()} ({platform.machine()}), {torch.get_num_threads()} ядра
Прогонов: {RUNS} (+{WARMUP} разогрев)

3 канала (цвет):  {a3 * 1000:6.1f} мс | {1 / a3:5.2f} FPS
1 канал  (серый): {a1 * 1000:6.1f} мс | {1 / a1:5.2f} FPS

(старый цветной замер тем же методом был ~2.7 FPS — для сверки)
"""

print(report)
Path("results/benchmark_gray_cpu.txt").write_text(report, encoding="utf-8")
