import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from labels import *
import random
from PIL import Image

COLOR_DICT = [[255, 0, 0],
              [100, 100, 100],
              [0, 0, 255],
              [255, 255, 0],
              [0, 255, 0]]


def read_dir(path):
    allname = os.listdir(path)
    allTestDataName = []
    for i in range(len(allname)):
        allTestDataName.append(path + '\\' + allname[i])
    samples = random.sample(allTestDataName, 100)
    return samples


def saveResult(pre_path, pre_img):
    cv2.imwrite(pre_path, pre_img)


def main():
    path = r'C:\Users\86135\Desktop\train10000\datas\train10000\train_image'
    save_mask_path = r'C:\Users\86135\Desktop\Python_segmatation\tdf\maskresults\\'
    save_preimg_path = r'C:\Users\86135\Desktop\Python_segmatation\tdf\imgresults\\'
    k_path = read_dir(path)
    for i in range(len(k_path)):
        print(k_path[i])
        origin_img = cv2.imread(k_path[i])
        origin_img = cv2.resize(origin_img, (512, 512))
        grass_mask = grass(k_path[i])
        road_mask = road(k_path[i])
        red_house_mask = red_house(k_path[i])
        blue_house_mask = blue_house(k_path[i])
        hill_mask = hill(k_path[i])
        green_mask = green_plant(k_path[i])
        water_mask = water(k_path[i])
        mask_img = np.zeros(shape=(512, 512), dtype=np.uint8)
        mask_img[red_house_mask == 1] = 1                           # house==>1
        mask_img[(mask_img == 0) & (blue_house_mask == 1)] = 1      # house==>1
        mask_img[(mask_img == 0) & (road_mask == 1)] = 2            # road==>2
        mask_img[(mask_img == 0) & (water_mask == 1)] = 3           # water==>3
        mask_img[(mask_img == 0) & (hill_mask == 1)] = 4            # hill==>4
        mask_img[(mask_img == 0) & (grass_mask == 1)] = 5           # grass==>5
        mask_img[(mask_img == 0) & (green_mask == 1)] = 5           # green==>5
        classify_img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
        for N in range(5):
            classify_img[mask_img == N+1] = COLOR_DICT[N]
        saveResult(save_mask_path + str(i) + '.png', classify_img)
        origin_img = origin_img / 255
        classify_img = classify_img / 255
        origin_img = (origin_img * 200 + classify_img * 55).astype(np.uint8)
        saveResult(save_preimg_path + str(i) + '.png', origin_img)


if __name__ == '__main__':
    main()
