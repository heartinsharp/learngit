import os
from datetime import datetime
import xlwt
from keras.callbacks import ModelCheckpoint
from data import *
from model import *
import tensorflow as tf
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.config
print(tf.__version__)
gpus = config.experimental.list_physical_devices(device_type='GPU')
cpus = config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

'''
数据集相关参数
'''
#  训练数据图像路径
train_image_path = "/home/ymh/桌面/U-net/data/train_image"
#  训练数据标签路径
train_label_path = "/home/ymh/桌面/U-net/data/train_label"
#  验证数据图像路径
validation_image_path = "/home/ymh/桌面/U-net/data/val_image"
#  验证数据标签路径
validation_label_path = "/home/ymh/桌面/U-net/data/val_label"

'''
模型相关参数
'''
batch_size = 4  # 批大小
classNum = 8  # 类的数目(包括背景)
input_size = (256, 256, 3)  #  模型输入图像大小
epochs = 20  # 训练模型的迭代总轮数
learning_rate = 1e-4  # 初始学习率
premodel_path = 'unet_model.hdf5'  # 预训练模型地址
model_path = "unet_model_v2.hdf5"  # 训练模型保存地址
resize_shape = 256  # resize大小
train_num = len(os.listdir(train_image_path))  # 训练数据数目
validation_num = len(os.listdir(validation_image_path))  # 验证数据数目
steps_per_epoch = train_num / batch_size  # 训练集每个epoch有多少个batch_size
validation_steps = validation_num / batch_size  # 验证集每个epoch有多少个batch_size
# 标签的颜色字典,用于onehot编码
colorDict_GRAY = [100, 200, 300, 400, 500, 600, 700, 800]
colorDict_RGB = [[0, 0, 100],
                 [0, 0, 200],
                 [0, 100, 0],
                 [0, 200, 0],
                 [100, 0, 0],
                 [200, 0, 0],
                 [100, 100, 100],
                 [200, 200, 200]]

#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 train_image_path,
                                 train_label_path,
                                 classNum,
                                 colorDict_GRAY)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY)
#  定义模型
model = unet(pretrained_weights=premodel_path,
             input_size=input_size,
             classNum=classNum,
             learning_rate=learning_rate)
#  打印模型结构
model.summary()
#  回调函数
model_checkpoint = ModelCheckpoint(model_path,
                                   monitor='loss',
                                   verbose=1,  # 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                   save_best_only=True)

#  获取当前时间
start_time = datetime.now()

#  模型训练
history = model.fit_generator(train_Generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              callbacks=[model_checkpoint],
                              validation_data=validation_data,
                              validation_steps=validation_steps)

#  训练总时间
end_time = datetime.now()
log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
print(log_time)
with open('TrainTime.txt', 'w') as f:
    f.write(log_time)

#  保存并绘制loss,acc
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
for i in range(len(acc)):
    sheet.write(i, 0, acc[i])
    sheet.write(i, 1, val_acc[i])
    sheet.write(i, 2, loss[i])
    sheet.write(i, 3, val_loss[i])
book.save(r'AccAndLoss.xls')
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("accuracy.png", dpi=300)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("loss.png", dpi=300)
plt.show()
