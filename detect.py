"""
detect.py — распознавание объектов на одной картинке с помощью YOLOv11 (nano).

Код пишется на маке в PyCharm, а ВЫПОЛНЯЕТСЯ на Raspberry Pi 5 (Remote Python).
Берёт картинку из images/, прогоняет через модель и сохраняет результат
с нарисованными рамками в results/.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

# --- Настройки --
INPUT = "images/input.jpg"      # какую картинку распознаём
OUTPUT = "results/input.jpg"    # куда сохранить картинку с рамками
MODEL = "yolo11n.pt"            # модель YOLOv11 nano (самая лёгкая, под CPU Pi)

# Папку results создаём, если её ещё нет
Path("results").mkdir(exist_ok=True)

# 1. Загружаем предобученную модель (скачается автоматически при первом запуске)
model = YOLO(MODEL)

# 2. Распознаём объекты на картинке
results = model.predict(source=INPUT, verbose=False)
r = results[0]

# 3. Рисуем рамки и сохраняем результат на диск
annotated = r.plot()            # картинка с рамками как массив пикселей
cv2.imwrite(OUTPUT, annotated)

# 4. Печатаем человеческим языком, что нашли
print(f"Обработано:  {INPUT}")
print(f"Результат:   {OUTPUT}")
print(f"Найдено объектов: {len(r.boxes)}")
for box in r.boxes:
    name = r.names[int(box.cls[0])]     # название: person, bus, car, ...
    conf = float(box.conf[0])           # уверенность модели, 0..1
    print(f"  - {name}: {conf:.0%}")
