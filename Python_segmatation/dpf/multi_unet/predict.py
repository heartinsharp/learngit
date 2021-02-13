from data import *
from model import *
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model
from keras import Model

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
path = '/home/ymh/桌面/U-net/data/test_image'
pre_test_path = '/home/ymh/桌面/U-net/data/pre_rgb'
model_path = 'unet_model.hdf5'
colorDict_GRAY = [100, 200, 300, 400, 500, 600, 700, 800]
colorDict_RGB = [[0, 0, 100],
                 [0, 0, 200],
                 [0, 100, 0],
                 [0, 200, 0],
                 [100, 0, 0],
                 [200, 0, 0],
                 [100, 100, 100],
                 [200, 200, 200]]

test_data = testGenerator(path, resize_shape=None)
model = load_model(model_path)
result = model.predict(test_data)
saveResult(path, pre_test_path, result, np.array(colorDict_RGB), output_size=(256, 256))
print("complete!")
