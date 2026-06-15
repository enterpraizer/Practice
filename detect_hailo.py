"""Распознавание объектов на картинке с помощью YOLO на ускорителе Hailo-8L (DeGirum)."""

from pathlib import Path

import degirum as dg

INPUT = "images/input.jpg"
OUTPUT = "results/hailo.jpg"
MODEL = "yolov8s_coco--320x320_quant_hailort_hailo8l_1"

Path("results").mkdir(exist_ok=True)
token = Path("degirum_token.txt").read_text().strip()

model = dg.load_model(
    model_name=MODEL,
    inference_host_address="@local",
    zoo_url="degirum/hailo",
    token=token,
    device_type="HAILORT/HAILO8L",
)

result = model(INPUT)

overlay = result.image_overlay
try:
    overlay.save(OUTPUT)            # PIL-изображение
except AttributeError:
    import cv2
    cv2.imwrite(OUTPUT, overlay)   # numpy-массив

print(f"Обработано:  {INPUT}")
print(f"Результат:   {OUTPUT}")
print(f"Найдено объектов: {len(result.results)}")
for det in result.results:
    print(f"  - {det['label']}: {det['score']:.0%}")
