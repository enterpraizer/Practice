"""Детекция на ч/б картинке (CPU, YOLOv11n): рамки + список объектов."""

from pathlib import Path

import cv2
from ultralytics import YOLO

INPUT = "images/input_gray.jpg"
OUTPUT = "results/gray.jpg"
MODEL = "yolo11n.pt"

Path("results").mkdir(exist_ok=True)

model = YOLO(MODEL)
results = model.predict(source=INPUT, verbose=False)
r = results[0]

cv2.imwrite(OUTPUT, r.plot())

print(f"Обработано:  {INPUT}")
print(f"Результат:   {OUTPUT}")
print(f"Найдено объектов: {len(r.boxes)}")
for box in r.boxes:
    name = r.names[int(box.cls[0])]
    conf = float(box.conf[0])
    print(f"  - {name}: {conf:.0%}")
