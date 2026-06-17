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

## Как модель попала на ускоритель

Hailo понимает только формат `.hef`, поэтому ту же `yolo11n.pt` пришлось скомпилировать
(на x86-Linux через Hailo Dataflow Compiler + Model Zoo):

```
yolo11n.pt  →  ONNX  →  квантизация в int8 (калибровка на COCO)  →  yolov11n.hef (hailo8l)
```

## Установка

```bash
# Система + окружение
sudo apt install -y python3-pip python3-venv libgl1
python3 -m venv yolo && source yolo/bin/activate

# CPU: PyTorch обязательно из CPU-репозитория (иначе тянет ненужные CUDA-пакеты)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics

# Hailo: драйвер + рантайм, затем библиотека для запуска из Python
sudo apt install -y hailo-all && sudo reboot
pip install degirum degirum_tools
degirum token install --token <ВАШ_ТОКЕН_DEGIRUM>
```

## Запуск

```bash
python3 detect.py            # детекция на CPU (YOLOv11n)
python3 benchmark.py         # замер скорости (CPU)
python3 detect_hailo.py      # YOLOv11n на Hailo: рамки + замер FPS
```

## Файлы

`detect*.py` — детекция на картинке, `benchmark*.py` — замер FPS, `*_hailo*` — версии
под ускоритель, `yolov11n.hef` — наша модель, скомпилированная под Hailo.
Исходники в `images/`, результаты в `results/`.
