import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random


def readfile(dir):
    allName = os.listdir(dir)
    allTestDataName = []
    for filename in os.listdir(dir):
        if (filename.endswith('.png') or filename.endswith('.tif')) and not ('pre' in filename):  # 文件名中不包含'pre'字符串
            allTestDataName.append(filename)
    allTestDataName.sort(key=lambda x: int(x[:-4]))
    return allTestDataName


def readTif(dir):
    return cv2.imread(dir, flags=-1)


def dataPreprocess(img, label, classNum, colorDict_GRAY):
    img = img / 255.0
    img = img.astype(np.float32)
    for i in range(len(colorDict_GRAY)):
        label[label == colorDict_GRAY[i]] = i
    new_label = np.zeros(label.shape + (classNum,))
    for i in range(classNum):
        new_label[label == i, i] = 1
    label = new_label.astype(np.float32)
    return img, label


def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath, -1)
        print(ImagePath)
        plt.imshow(img)
        plt.show()
        img_new = img
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        print(colorDict)
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if len(colorDict) == classNum:
            break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        color_RGB = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_RGB.append(color_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1, colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY


def trainGenerator(batch_size, train_image_path, train_label_path, classNum, colorDict_GRAY, resize_shape=None):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    imageList = np.unique(imageList)
    labelList = np.unique(labelList)
    img = readTif(train_image_path + "/" + imageList[0])
    #  无限生成数据
    while True:
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.float32)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.float32)
        if resize_shape is not None:
            img_generator = np.zeros((batch_size, resize_shape, resize_shape, resize_shape), np.float32)
            label_generator = np.zeros((batch_size, resize_shape, resize_shape), np.float32)
        #  随机生成一个batch的起点
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            img = readTif(train_image_path + "/" + imageList[rand + j])
            #  改变图像尺寸至特定尺寸(
            #  因为resize用的不多，我就用了OpenCV实现的，这个不支持多波段，需要的话可以用np进行resize
            if resize_shape is not None:
                img = cv2.resize(img, (resize_shape, resize_shape))
            img_generator[j] = img
            label = readTif(train_label_path + "/" + labelList[rand + j])
            #  若为彩色，转为灰度
            if len(label.shape) == 3:
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if resize_shape is not None:
                label = cv2.resize(label, (resize_shape, resize_shape))
            label_generator[j] = label

        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, colorDict_GRAY)

        yield img_generator, label_generator


def testGenerator(test_iamge_path, resize_shape=None):
    imageList = os.listdir(test_iamge_path)
    for i in range(len(imageList)):
        img = readTif(test_iamge_path + "/" + imageList[i])
        #  归一化
        img = img / 255.0
        img = img.astype(np.float32)
        if resize_shape is not None:
            #  改变图像尺寸至特定尺寸
            img = cv2.resize(img, (resize_shape, resize_shape))
        #  将测试图片扩展一个维度,与训练时的输入[batch_size,img.shape]保持一致
        img = np.reshape(img, (1,) + img.shape)
        yield img


def saveResult(test_image_path, test_predict_path, model_predict, color_dict, output_size):
    imageList = os.listdir(test_image_path)
    for i, img in enumerate(model_predict):
        channel_max = np.argmax(img, axis=-1)  # return channel_numbers
        img_out = np.array(color_dict[channel_max], np.uint8)
        #  修改差值方式为最邻近差值
        img_out = cv2.resize(img_out, (output_size[0], output_size[1]), interpolation=cv2.INTER_NEAREST)
        #  保存为无损压缩png
        cv2.imwrite(test_predict_path + "/" + imageList[i][:-4] + ".png", img_out)


def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    #  返回所有类别的精确率precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return precision


def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU