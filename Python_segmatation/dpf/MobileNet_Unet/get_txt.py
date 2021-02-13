import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
path1 = '/home/ymh/桌面/U-net/mobilenet/datasets/img'
path2 = '/home/ymh/桌面/U-net/mobilenet/datasets/label'
# label = cv2.imread(path2, -1)
# print(label[540, 640])
# print(label.shape)


# get train.txt
lis1 = os.listdir(path1)
lis1 = np.unique(lis1)
print(lis1)


lis2 = os.listdir(path2)
lis2 = np.unique(lis2)
print(lis2)

f1 = open('/home/ymh/桌面/U-net/mobilenet/datasets/train.txt', 'w')
for i in range(len(lis1)):
    f1.write(lis1[i] + ";" + lis2[i] + "\n")
