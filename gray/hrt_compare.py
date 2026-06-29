"""Честное сравнение RGB (3 канала) vs grey (1 канал) YOLOv11n на Hailo-8L.

Оба прогона — ОДНИМ методом, напрямую через HailoRT (без DeGirum): на реальной картинке,
с препроцессингом + инференсом + разбором NMS в каждом кадре (end-to-end на картинке).
"""

import time

import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                            InputVStreamParams, OutputVStreamParams, InferVStreams,
                            FormatType)

IMG = "/home/nikita/yolo-pi/rgb/images/input.jpg"
SCORE_THR = 0.30
RUNS = 30
WARMUP = 5

MODELS = [
    ("RGB  (3 канала)", "/home/nikita/yolo-pi/rgb/yolov11n.hef", 3),
    ("grey (1 канал)",  "/home/nikita/yolo-pi/gray/yolov11n_gray.hef", 1),
]

orig = cv2.imread(IMG)


def prep(channels):
    if channels == 1:
        g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (640, 640))
        return g.reshape(640, 640, 1).astype(np.uint8)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (640, 640))
    return rgb.astype(np.uint8)


def count_objects(out_arr):
    det = out_arr[0]
    n = 0
    for arr in det:
        if arr is None or len(arr) == 0:
            continue
        for row in np.array(arr):
            if float(row[4]) >= SCORE_THR:
                n += 1
    return n


def run(name, hef_path, channels):
    hef = HEF(hef_path)
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()
        in_p = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        with InferVStreams(ng, in_p, out_p) as pipe:
            with ng.activate(ng_params):
                for _ in range(WARMUP):
                    pipe.infer({in_info.name: np.expand_dims(prep(channels), 0)})
                times = []
                for _ in range(RUNS):
                    s = time.perf_counter()
                    inp = np.expand_dims(prep(channels), 0)          # препроцессинг
                    res = pipe.infer({in_info.name: inp})            # инференс
                    n = count_objects(res[out_info.name])            # разбор NMS
                    times.append(time.perf_counter() - s)
    avg = sum(times) / len(times)
    return avg, n


print(f"{'Модель':<18}{'мс/кадр':>10}{'FPS':>10}{'объектов':>12}")
print("-" * 50)
rows = []
for name, path, ch in MODELS:
    avg, n = run(name, path, ch)
    rows.append((name, avg, n))
    print(f"{name:<18}{avg*1000:>10.1f}{1/avg:>10.1f}{n:>12}")

if len(rows) == 2:
    speedup = rows[0][1] / rows[1][1]
    print("-" * 50)
    print(f"grey быстрее RGB в {speedup:.2f}x (end-to-end на картинке, HailoRT)")
