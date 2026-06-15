"""Распознавание объектов на картинке с помощью YOLOv11 (nano) на Raspberry Pi 5."""

from pathlib import Path

import cv2
from ultralytics import YOLO

INPUT = "images/input.jpg"
OUTPUT = "results/input.jpg"
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
