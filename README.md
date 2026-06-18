# YOLO на Raspberry Pi 5: CPU vs Hailo

Запустил **YOLOv11n** на Raspberry Pi 5 двумя способами — на **CPU** и на AI-ускорителе
**Hailo-8L** — и сравнил скорость на **одной и той же модели** (640×640).
Модель для ускорителя скомпилировал сам из `yolo11n.pt` (квантизация в int8 на картинках COCO).

**Итог: на ускорителе ~12× быстрее (2.7 → 33.5 FPS).**

## Железо

- Raspberry Pi 5, 8 GB
- AI-ускоритель Hailo-8L (Raspberry Pi AI Kit, 13 TOPS)
- Без камеры — распознавание на готовых картинках

## Результат

Одно фото (зимняя пробка), **YOLOv11n, 640×640**:

| | CPU | Hailo-8L |
|---|---|---|
| Скорость | 2.7 FPS (371 мс) | **33.5 FPS (30 мс)** |
| Найдено объектов | 21 | 19 |
| Ускорение | 1× | **≈ 12×** |

| CPU | Hailo |
|---|---|
| ![CPU](results/input.jpg) | ![Hailo](results/hailo.jpg) |

Небольшая разница в числе объектов (21 vs 19) — из-за квантизации модели в int8.

Замеры: [results/benchmark.txt](results/benchmark.txt) (CPU),
[results/benchmark_hailo.txt](results/benchmark_hailo.txt) (Hailo).

## Компиляция модели под Hailo (квантизация)

Hailo выполняет модели только в формате `.hef` и считает в **int8**, поэтому `yolo11n.pt`
скомпилировал сам. Компиляция идёт на **x86-Linux** (использовал Google Colab — Pi и Mac не
подходят, нужен x86).

Пайплайн:
```
yolo11n.pt → ONNX → квантизация в int8 (калибровка на COCO128) → yolov11n.hef (hailo8l)
```

Окружение (под HailoRT 4.23, что стоит на плате):
- Python 3.10, Hailo Dataflow Compiler 3.33.1, Hailo Model Zoo 2.17

Шаги:
```bash
# 1. калибровочные картинки (нужны для квантизации)
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
unzip coco128.zip

# 2. parse → квантизация в int8 на COCO → compile  (одной командой)
hailomz compile yolov11n --hw-arch hailo8l --calib-path coco128/images/train2017
# результат: yolov11n.hef
```

**Что значит квантизация:** числа модели переводятся из float32 в int8 (256 уровней). Чтобы
подобрать масштаб без потери точности, через модель прогоняют картинки COCO и замеряют
реальные диапазоны значений в каждом слое. Отсюда небольшая разница в детекциях (19 vs 21) —
плата за скорость.

Запуск `.hef` из Python — через DeGirum (локальная модель: `.hef` + JSON-конфиг с
`OutputPostprocessType: DetectionYoloHailo`), см. `detect_hailo.py`.

## Файлы

`detect*.py` — детекция на картинке, `benchmark*.py` — замер FPS, `*_hailo*` — версии
под ускоритель, `yolov11n.hef` — наша модель, скомпилированная под Hailo.
Исходники в `images/`, результаты в `results/`.
