from nets.unet import mobilenet_unet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
# import variance as var
import os
import cv2
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
config = tf.config
print(tf.__version__)
gpus = config.experimental.list_physical_devices(device_type='GPU')
cpus = config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
)

NCLASSES = 10
HEIGHT = 256  # TODO
WIDTH = 256  # TODO


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread("./datasets/image" + '/' + name, 1)
            img = np.array(img)
            img = img / 255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            img = cv2.imread("./datasets/label" + '/' + name, 0)
            img = cv2.resize(img, (int(WIDTH // 2), int(HEIGHT // 2)))
            # img = img.resize((int(WIDTH / 2), int(HEIGHT / 2)))
            seg_labels = np.zeros((WIDTH // 2, HEIGHT // 2, NCLASSES), np.uint8)
            for c in range(NCLASSES):
                seg_labels[:, :, c] = (img[:, :] == c).astype(int)
            seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i + 1) % n
        yield np.array(X_train), np.array(Y_train)


# def my_loss(y_true, y_pred):
#     crossloss = K.categorical_crossentropy(y_true, y_pred)
#     vari = var.variance(y_pred)
#     loss = (K.sum(crossloss) / HEIGHT / WIDTH + 0.5 * K.sum(vari) / HEIGHT / WIDTH / NCLASSES) / 16
#     return loss


if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = mobilenet_unet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    # BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
    #                     'releases/download/v0.6/')
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)

    # weight_path = BASE_WEIGHT_PATH + model_name
    # weights_path = keras.utils.get_file(model_name, weight_path)
    # print(weight_path)
    # model.load_weights(weights_path, by_name=True, skip_mismatch=True)  # 模型权重

    # 打开数据集的txt
    with open("./datasets/train.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 保存的方式，1世代保存一次
    checkpoint_period = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=1
    )

    # 学习率下降的方式，val_loss三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    batch_size = 16
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    Tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1, write_images=True)
    # 开始训练
    model._get_distribution_strategy = lambda: None
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period, reduce_lr, Tensorboard])

    model.save_weights(log_dir + 'last1.h5')


