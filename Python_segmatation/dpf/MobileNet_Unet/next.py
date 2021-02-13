import os
import cv2
import random

test_list = os.listdir('./datasets/img')
a = random.sample(test_list, 50)
for i in range(len(a)):
    img = cv2.imread('./datasets/img/' + a[i])
    cv2.imwrite('./img/' + a[i][:-4] + '.png', img)



