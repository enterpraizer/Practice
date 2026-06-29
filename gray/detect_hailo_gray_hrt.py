"""Детекция grey (1-канальной) YOLOv11n на ускорителе Hailo-8L НАПРЯМУЮ через HailoRT.

DeGirum в установленной версии не умеет подавать 1-канальный вход, поэтому работаем
с HailoRT напрямую: сами формируем тензор (640x640x1, uint8) и получаем детекции
со встроенного на чипе NMS. Рисуем рамки и меряем end-to-end FPS на реальной картинке.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                            InputVStreamParams, OutputVStreamParams, InferVStreams,
                            FormatType)

HEF_PATH = "/home/nikita/yolo-pi/gray/yolov11n_gray.hef"
IMG = "/home/nikita/yolo-pi/rgb/images/input.jpg"
OUT = "/home/nikita/yolo-pi/gray/results/hailo_gray.jpg"
SCORE_THR = 0.30
RUNS = 20
WARMUP = 3

COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag',
'tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table',
'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
'hair drier','toothbrush']

Path(OUT).parent.mkdir(parents=True, exist_ok=True)

hef = HEF(HEF_PATH)
in_info = hef.get_input_vstream_infos()[0]
out_info = hef.get_output_vstream_infos()[0]
H, W, C = in_info.shape
print(f"Вход модели: {in_info.shape} ({in_info.name})")

# готовим серый вход (640x640x1, uint8)
orig = cv2.imread(IMG)
oh, ow = orig.shape[:2]
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
gray_r = cv2.resize(gray, (W, H))
inp = gray_r.reshape(H, W, 1).astype(np.uint8)
batch = np.expand_dims(inp, axis=0)

with VDevice() as target:
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = target.configure(hef, cfg)[0]
    ng_params = ng.create_params()
    in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    with InferVStreams(ng, in_params, out_params) as pipe:
        with ng.activate(ng_params):
            for _ in range(WARMUP):
                pipe.infer({in_info.name: batch})
            times = []
            for _ in range(RUNS):
                s = time.perf_counter()
                results = pipe.infer({in_info.name: batch})
                times.append(time.perf_counter() - s)

avg = sum(times) / len(times)
out = results[out_info.name]

# диагностика формата выхода NMS
print("Тип выхода:", type(out))
det = out[0] if isinstance(out, (list, np.ndarray)) else out
print("Классов в выходе:", len(det))

# парсинг NMS-by-class: список из 80 массивов Nx5 [ymin,xmin,ymax,xmax,score] (норм. 0..1)
boxes = []
for cls_id, arr in enumerate(det):
    if arr is None or len(arr) == 0:
        continue
    a = np.array(arr)
    for row in a:
        ymin, xmin, ymax, xmax, score = [float(v) for v in row[:5]]
        if score < SCORE_THR:
            continue
        boxes.append((cls_id, score, xmin, ymin, xmax, ymax))

print(f"\nНайдено объектов (score>{SCORE_THR}): {len(boxes)}")
for cls_id, score, *_ in sorted(boxes, key=lambda b: -b[1]):
    name = COCO[cls_id] if cls_id < len(COCO) else str(cls_id)
    print(f"  - {name}: {score:.0%}")

# рисуем рамки на СЕРОЙ картинке (то, что реально видит модель),
# переводим в 3 канала только чтобы рамки/подписи были цветными
canvas = cv2.cvtColor(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
for cls_id, score, xmin, ymin, xmax, ymax in boxes:
    x1, y1, x2, y2 = int(xmin*ow), int(ymin*oh), int(xmax*ow), int(ymax*oh)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    name = COCO[cls_id] if cls_id < len(COCO) else str(cls_id)
    cv2.putText(canvas, f"{name} {score:.0%}", (x1, max(0, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite(OUT, canvas)

print(f"\nРезультат сохранён: {OUT}")
print(f"Среднее время на картинку: {avg*1000:.1f} мс")
print(f"Производительность (end-to-end): {1/avg:.2f} FPS")
