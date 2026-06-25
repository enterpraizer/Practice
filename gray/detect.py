"""Детекция на ч/б картинке (CPU, YOLOv11n): рамки + список объектов."""

from pathlib import Path

import cv2
from ultralytics import YOLO

SRC = "images/input.jpg"
GRAY = "images/input_gray.jpg"
OUTPUT = "results/gray.jpg"
MODEL = "yolo11n.pt"

Path("results").mkdir(exist_ok=True)

# делаем ч/б версию из цветной
img = cv2.imread(SRC)
cv2.imwrite(GRAY, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

model = YOLO(MODEL)
results = model.predict(source=GRAY, verbose=False)
r = results[0]

cv2.imwrite(OUTPUT, r.plot())

print(f"Обработано:  {GRAY}")
print(f"Результат:   {OUTPUT}")
print(f"Найдено объектов: {len(r.boxes)}")
for box in r.boxes:
    name = r.names[int(box.cls[0])]
    conf = float(box.conf[0])
    print(f"  - {name}: {conf:.0%}")
