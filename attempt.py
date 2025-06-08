import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to image.jpg
image_path = os.path.join(script_dir, 'image.jpg')

# Check if the file exists
if os.path.exists(image_path):
    print(f"File found at: {image_path}")
else:
    print(f"File not found at: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("❌ Помилка: Зображення не знайдено. Перевір шлях до файлу.")
    exit()

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

edge_x = cv2.filter2D(image, -1, sobel_x)
edge_y = cv2.filter2D(image, -1, sobel_y)

edges = cv2.magnitude(np.float32(edge_x), np.float32(edge_y))

plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Оригінал')
plt.subplot(1, 3, 2), plt.imshow(edge_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(1, 3, 3), plt.imshow(edges, cmap='gray'), plt.title('Границі')
plt.show()
