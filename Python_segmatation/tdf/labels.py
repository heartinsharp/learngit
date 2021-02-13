import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure, morphology
import math


def grass(path):
    img1 = cv2.imread(path, flags=-1)  # flags=-1 读取原图
    img1 = img1[:, :, [2, 1, 0]]  # 图像通道由BGR转为RGB
    img2 = cv2.resize(img1, (img1.shape[0] * 2, img1.shape[1] * 2))  # 调整图像的大小（shape）
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    mask = np.zeros(img_gray.shape, dtype=img_gray.dtype)  # 创建一个灰度图承接最终结果
    # inds.shape=256X256,inds.dtype=boolean,type(inds)=np.ndarray
    inds11, inds12 = img2[:, :, 0] <= 100, img2[:, :, 0] >= 80  # 通道为R
    inds21, inds22 = img2[:, :, 1] <= 120, img2[:, :, 1] >= 80  # 通道为G
    inds31, inds32 = img2[:, :, 2] <= 90, img2[:, :, 2] >= 70  # 通道为B
    inds = inds11 & inds21 & inds31 & inds12 & inds22 & inds32  # 取上面得到的bool矩阵的交集
    mask[inds] = 1  # mask上符合要求的所有坐标置1

    return mask


def water(path):
    img1 = cv2.imread(path)
    img2 = cv2.resize(img1, (img1.shape[0] * 2, img1.shape[1] * 2))
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_grb = img2[:, :, [1, 2, 0]]  # 图像通道由BGR转为RGB
    # step1
    # 图像三通道两两相减再加上15后的差值在[0，30]初步判断为水
    img_sub = img_grb - img2 + 15
    img_sub_gray = cv2.cvtColor(img_sub, cv2.COLOR_BGR2GRAY)
    inds00 = img_sub_gray <= 30
    # step2
    # 对原图的灰度图像素设置阈值为[30,70]间的进一步判断是否为水
    inds11, inds22 = img_gray <= 70, img_gray >= 30
    # 对一二两步所得的bool型矩阵进行取交集
    inds = inds00 & inds11 & inds22
    mask = np.zeros(img_gray.shape, dtype=img_gray.dtype)
    mask[inds] = 1
    # step3
    # 去除部分面积较小的噪声
    mask = cv2.blur(mask, (7, 7))
    mask = morphology.dilation(mask, np.ones([5, 5]))   # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))    # 形态学腐蚀操作
    # 对每一个连通域进行面积取舍并赋给新图
    labels = measure.label(mask, connectivity=2)
    label_vals = np.unique(labels)
    rst = np.zeros(mask.shape, dtype=mask.dtype)
    for N in label_vals:
        if N == 0:
            continue
        else:
            num = np.sum(labels == N)
            if num >= 20000:
                rst = rst + np.where(labels == N, 1, 0)
    # 返回最终result的mask图，标签为2
    return rst


def red_house(path):
    img1 = cv2.imread(path)
    img2 = cv2.resize(img1, (img1.shape[0] * 2, img1.shape[1] * 2))
    img2 = cv2.blur(img2, (7, 7))
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    inds1, inds2, inds3 = img2[:, :, 0] <= 75, img2[:, :, 1] <= 85, img2[:, :, 2] >= 95
    inds = inds1 & inds2 & inds3
    mask = np.zeros(img_gray.shape, dtype=img_gray.dtype)
    mask[inds] = 1
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    labels = measure.label(mask, connectivity=2)
    label_vals = np.unique(labels)
    rst = np.zeros(mask.shape, dtype=mask.dtype)
    for N in label_vals:
        if N == 0:
            continue
        else:
            num = np.sum(labels == N)
            if num >= 100:
                rst = rst + np.where(labels == N, 1, 0)
    # 返回最终result的mask图，标签为2
    return rst


def blue_house(path):
    img1 = cv2.imread(path)
    img2 = cv2.resize(img1, (img1.shape[0] * 2, img1.shape[1] * 2))
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.blur(img2, (7, 7))
    inds1, inds2, inds3 = img2[:, :, 0] <= 230, img2[:, :, 1] <= 170, img2[:, :, 2] <= 120
    inds11, inds12 = img2[:, :, 0] >= 140, img2[:, :, 2] >= 50
    inds = inds1 & inds2 & inds3 & inds11 & inds12
    mask = np.zeros(img_gray.shape, dtype=img_gray.dtype)
    mask[inds] = 1
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    labels = measure.label(mask, connectivity=2)
    label_vals = np.unique(labels)
    rst = np.zeros(mask.shape, dtype=mask.dtype)
    for N in label_vals:
        if N == 0:
            continue
        else:
            num = np.sum(labels == N)
            if num >= 100:
                rst = rst + np.where(labels == N, 1, 0)
    # 返回最终result的mask图，标签为2
    return rst


def road(path):
    img1 = cv2.imread(path)
    img2 = cv2.resize(img1, (img1.shape[0] * 2, img1.shape[1] * 2))
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.blur(img_gray, (7, 7))
    mask = np.zeros(img_gray.shape, img_gray.dtype)
    inds = img2 >= 120
    mask[inds] = 1
    mask = morphology.dilation(mask, np.ones([9, 9]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([9, 9]))  # 形态学腐蚀操作
    mask = morphology.dilation(mask, np.ones([3, 3]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([3, 3]))  # 形态学腐蚀操作
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    labels = measure.label(mask, connectivity=2)
    label_vals = np.unique(labels)
    rst = np.zeros(mask.shape, dtype=mask.dtype)
    for N in label_vals:
        if N == 0:
            continue
        else:
            num = np.sum(labels == N)
            if 1400 <= num <= 3500 or num >= 14000:
                rst = rst + np.where(labels == N, 1, 0)
    # 返回最终result的mask图，标签为2
    return rst


def hill(path):
    img1 = cv2.imread(path)
    img2 = cv2.resize(img1, (img1.shape[0] * 2, img1.shape[1] * 2))
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.blur(img2, (7, 7))
    mask = np.zeros(img_gray.shape, img_gray.dtype)
    inds11, inds12 = img2[:, :, 0] <= 180, img2[:, :, 0] >= 100  # 通道为R
    inds21, inds22 = img2[:, :, 1] <= 250, img2[:, :, 1] >= 140  # 通道为G
    inds31, inds32 = img2[:, :, 2] <= 255, img2[:, :, 2] >= 160  # 通道为B
    inds = inds11 & inds21 & inds31 & inds12 & inds22 & inds32  # 取上面得到的bool矩阵的交集
    mask[inds] = 1  # mask上符合要求的所有坐标置1
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    return mask


def green_plant(path):
    img1 = cv2.imread(path)
    img2 = cv2.resize(img1, (img1.shape[0] * 2, img1.shape[1] * 2))
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.blur(img2, (7, 7))
    mask = np.zeros(img_gray.shape, img_gray.dtype)
    inds11, inds12 = img2[:, :, 0] <= 80, img2[:, :, 0] >= 40  # 通道为R
    inds21, inds22 = img2[:, :, 1] >= 70, img2[:, :, 1] >= 50  # 通道为G
    inds31, inds32 = img2[:, :, 2] <= 90, img2[:, :, 2] >= 40  # 通道为B
    inds = inds11 & inds21 & inds31 & inds12 & inds22 & inds32  # 取上面得到的bool矩阵的交集
    mask[inds] = 1  # mask上符合要求的所有坐标置1
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.dilation(mask, np.ones([5, 5]))  # 形态学膨胀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    mask = morphology.erosion(mask, np.ones([5, 5]))  # 形态学腐蚀操作
    return mask


# path = r'C:\Users\86135\Desktop\train10000\datas\train10000\train_image\675.png'
# img = cv2.imread(path, flags=-1)
# img_rgb = img[:, :, [2, 1, 0]]
# water_mask = green_plant(path)
# plt.imshow(water_mask)
# plt.show()
# plt.imshow(img)
# plt.show()
# plt.imshow(img_rgb)
# plt.show()
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
