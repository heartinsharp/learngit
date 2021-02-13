import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def readfile(path):
    allName = os.listdir(path)
    allTestDataName = []
    for filename in os.listdir(path):
        if (filename.endswith('.png') or filename.endswith('.tif')) and not ('pre' in filename):  # 文件名中不包含'pre'字符串
            allTestDataName.append(filename)
    allTestDataName.sort(key=lambda x: int(x[:-4]))
    return allTestDataName


img_path = '/home/ymh/桌面/U-net/data/trash/ori_image'
label_path = '/home/ymh/桌面/U-net/data/trash/ori_label'
test_img_path = '/home/ymh/桌面/U-net/mobilenet/img'


img_list = readfile(img_path)
print(len(img_list))
a = random.sample(img_list, 50)
a = np.unique(a)
b = list(set(img_list) - set(a))
b = np.unique(b)
print(len(b))
print(len(a))
num = 0
for i in range(len(a)):
    img = cv2.imread(img_path + "/" + a[i])
    label = cv2.imread(label_path + "/" + a[i].split(".", 1)[0] + '.png', -1)
    lis = np.unique(label)
    if 100 in lis:
        num += 1
        cv2.imwrite(test_img_path + '/' + a[i], img)
print(num)
print("complete!")
# for i in range(len(b)):
#     img = cv2.imread(img_path + "/" + b[i], -1)
#     label = cv2.imread(label_path + "/" + b[i].split(".", 1)[0] + '.png', -1)
#     cv2.imwrite(train_img_path + '/' + b[i], img)
#     cv2.imwrite(train_label_path + "/" + b[i].split(".", 1)[0] + '.png', label)
# print("complete!")



