import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

path = r'C:\Users\86135\Desktop\train10000\datas\train10000\train_image\69.png'
img = cv2.imread(path, -1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.resize(img, (512, 512))
img_rgb = img_rgb[:, :, [2, 1, 0]]  # bgr==>rbg

mask = np.zeros(img_gray.shape, dtype=img_gray.dtype)

plt.imshow(mask)
plt.show()
