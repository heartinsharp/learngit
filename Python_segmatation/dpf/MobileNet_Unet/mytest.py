import cv2
import numpy as np
import matplotlib.pyplot as plt
path = r'C:\Users\86135\Desktop\train10000\datas\train10000\train_label/423190.png'
img = cv2.imread(path, -1)
print(img.shape)
print(img.dtype)
plt.imshow(img*50)
plt.show()
