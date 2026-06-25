"""Сделать ч/б (1-канальную) версию картинки."""

import cv2

img = cv2.imread("images/input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("images/input_gray.jpg", gray)
print("Сохранено: images/input_gray.jpg (1 канал)")
