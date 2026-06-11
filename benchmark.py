"""
benchmark.py — замер скорости инференса YOLOv11 на CPU Raspberry Pi 5.

Прогоняет распознавание много раз на одной картинке и считает:
  - среднее время на картинку (мс)
  - производительность (FPS, кадров в секунду)
Первые прогоны идут "на разогрев" и в статистику не попадают.
Результат печатается в консоль и сохраняется в results/benchmark.txt.
"""

import platform
import time
from pathlib import Path

from ultralytics import YOLO

# --- Настройки ---
INPUT = "images/input.jpg"   # картинка для замера
MODEL = "yolo11n.pt"         # модель YOLOv11 nano
RUNS = 30                    # сколько прогонов считаем
WARMUP = 3                   # сколько прогонов "на разогрев" (не считаем)

Path("results").mkdir(exist_ok=True)

# Загружаем модель
model = YOLO(MODEL)

# Разогрев: самый первый инференс всегда медленнее, его не учитываем
for _ in range(WARMUP):
    model.predict(source=INPUT, verbose=False)

# Замер: засекаем время каждого прогона
times = []
for _ in range(RUNS):
    start = time.perf_counter()
    model.predict(source=INPUT, verbose=False)
    times.append(time.perf_counter() - start)

avg_s = sum(times) / len(times)     # среднее время, секунды
avg_ms = avg_s * 1000               # среднее время, миллисекунды
fps = 1 / avg_s                     # кадров в секунду

# Формируем отчёт
report = f"""Замер скорости YOLOv11n на Raspberry Pi 5 (CPU)
=================================================
Машина:        {platform.node()} ({platform.machine()})
Python:        {platform.python_version()}
Модель:        YOLOv11n (CPU)
Картинка:      {INPUT}
Прогонов:      {RUNS} (+{WARMUP} на разогрев)

Среднее время на картинку: {avg_ms:.1f} мс
Производительность:        {fps:.2f} FPS
Минимум / максимум:        {min(times) * 1000:.1f} / {max(times) * 1000:.1f} мс
"""

# Печатаем в консоль и сохраняем в файл
print(report)
Path("results/benchmark.txt").write_text(report, encoding="utf-8")
print("Отчёт сохранён в results/benchmark.txt")