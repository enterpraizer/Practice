"""Экспорт 1-канальной YOLOv11n в ONNX (для компиляции под Hailo).

Берём обычную yolo11n.pt, делаем вход 1-канальным (суммируем веса первого слоя),
экспортируем в ONNX. Этот ONNX потом компилируем в .hef в Colab.
"""

import onnx
import torch.nn as nn
from ultralytics import YOLO

m = YOLO("yolo11n.pt")

old = m.model.model[0].conv
new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
new.weight.data = old.weight.data.sum(dim=1, keepdim=True)
if old.bias is not None:
    new.bias.data = old.bias.data
m.model.model[0].conv = new
m.model.yaml["ch"] = 1
m.model.yaml["channels"] = 1

path = m.export(format="onnx", imgsz=640, opset=11)
print("ONNX сохранён:", path)

# проверка: какой вход у получившегося ONNX (должно быть 1 канал)
model = onnx.load(path)
shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
print("Форма входа ONNX:", shape)
