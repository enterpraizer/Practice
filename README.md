# YOLO на Raspberry Pi 5: CPU vs Hailo

Запустил детекцию объектов YOLO на Raspberry Pi 5 двумя способами — на **CPU** и на
AI-ускорителе **Hailo-8L** — и сравнил скорость при одинаковом разрешении 640×640.

**Итог: на ускорителе ~3.7× быстрее, и объектов находит больше.**

## Железо

- Raspberry Pi 5, 8 GB
- AI-ускоритель Hailo-8L (Raspberry Pi AI Kit, 13 TOPS)
- Без камеры — распознавание на готовых картинках

## Результат

Одно фото (зимняя пробка), вход 640×640:

| | CPU — YOLOv11n | Hailo — DAMO-YOLO |
|---|---|---|
| Скорость | 2.7 FPS (371 мс) | **10 FPS (100 мс)** |
| Найдено объектов | 21 | **35** |

| CPU | Hailo |
|---|---|
| ![CPU](results/input.jpg) | ![Hailo](results/hailo.jpg) |

Замеры: [results/benchmark.txt](results/benchmark.txt) (CPU),
[results/benchmark_hailo.txt](results/benchmark_hailo.txt) (Hailo).

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
python3 detect.py          # детекция на CPU
python3 benchmark.py       # замер скорости (CPU)
python3 detect_hailo.py    # детекция на Hailo
python3 benchmark_hailo.py # замер скорости (Hailo)
python3 video_hailo.py     # детекция на видео (Hailo)
```

## Файлы

`detect*.py` — детекция на картинке, `benchmark*.py` — замер FPS
(`*_hailo` — версии под ускоритель). Исходники в `images/`, результаты в `results/`.
