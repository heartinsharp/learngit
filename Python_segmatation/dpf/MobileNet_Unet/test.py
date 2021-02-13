import json
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random

random.seed(1)
path = r'D:\AliTianChiCiZhuanQueXian\tile_round1_train_20201231\train_imgs\\'
save_label_path = r'D:\AliTianChiCiZhuanQueXian\Code\datasets\label'
save_img_path = r'D:\AliTianChiCiZhuanQueXian\Code\datasets\img'
annos_path = r'D:\AliTianChiCiZhuanQueXian\tile_round1_train_20201231\train_annos.json'
with open(annos_path, 'r', encoding='utf8') as fp:
    data = json.load(fp)
    for t in range(len(data)):
        current_path = os.path.join(path, data[t]['name'])
        large_img = cv2.imread(current_path, 1)
        bbox = data[t]['bbox']
        large_label = np.zeros((data[t]['image_height'], data[t]['image_width']), large_img.dtype).astype(np.uint8)
        large_label[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])] = 1
        size_x, size_y = large_img.shape[0], large_img.shape[1]
        if bbox[2] <= 416 & bbox[3] <= 416:
            img = large_img[0:415, 0:415]
            label = large_label[0:415, 0:415]
        elif bbox[0] >= (size_x - 417) & bbox[3] <= 416:
            img = large_img[size_x - 417:size_x - 1, 0:415]
            label = large_label[size_x - 417:size_x - 1, 0:415]
        elif bbox[1] >= (size_y - 417) & bbox[2] <= 416:
            img = large_img[0:415, size_y - 417:size_y - 1]
            label = large_label[0:415, size_y - 417:size_y - 1]
        elif bbox[0] >= (size_x - 417) & bbox[1] >= (size_y - 417):
            img = large_img[size_x - 417:size_x - 1, size_y - 417:size_y - 1]
            label = large_label[size_x - 417:size_x - 1, size_y - 417:size_y - 1]
        else:
            rx = random.randint(0, 100)
            ry = random.randint(0, 100)
            img = large_img[bbox[0] - rx:bbox[0] - rx + 416, bbox[1] - rx:bbox[1] - rx + 416]
            label = large_label[bbox[0] - rx:bbox[0] - rx + 416, bbox[1] - rx:bbox[1] - rx + 416]
        if data[t]['category'] == 4:
            save_to_name = os.path.join(save_img_path, data[t]['name'])
            cv2.imwrite(save_to_name, img)
            save_to_name = os.path.join(save_label_path, data[t]['name'])
            cv2.imwrite(save_to_name, label)
        break

