# This is a sample Python script.
import os
import cv2
import numpy as np
import random
import tensorflow as tf

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class Transform(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, inputs, label):
        inputs = cv2.resize(inputs, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return inputs, label


class Data_loader(object):
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform,
                 shuffle=True):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle

        self.data_list = self.read_list()

    def read_list(self):
        data_list = []
        with open(self.image_list_file) as infile:
            for line in infile:
                data_path = os.path.join(self.image_folder, line.split()[0])
                label_path = os.path.join(self.image_folder, line.split()[1])
                data_list.append((data_path, label_path))

        random.shuffle(data_list)
        return data_list

    def preprocess(self, data, label):
        h, w, c = data.shape
        h_gt, w_gt = label.shape
        assert h == h_gt, "Error"
        assert w == w_gt, "Error"

        if self.transform:
            dta, label = self.transform(data, label)

        label = label[:, :, np.newaxis]  # [256, 256]==>[256, 256, 1]
        return data, label

    def __len__(self):
        return len(self.data_list)

    def __call__(self, *args, **kwargs):
        for data_path, label_path in self.data_list:
            data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            print(data.shape, label.shape)
            data, label = self.preprocess(data, label)

            yield data, label


def main():
    batch_size = 5
    # TODO
    '''
    place = fluid.CPUPlace()
    with fluid.dygraphy.guard(place):
        transform = Transform(256)
        # create Data_loader instance
        basic_dataloader = Data_loader(
            image_folder='',
            image_list_file='',
            transform=transform,
            shuffle=True
        )
    # create fluid.io.DataLoader instance
    datloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
    # set sample generator for fluid dataloader
    dataloader.set_sample_generator(basic_dataloader,
                                    batch_size=batch_size,
                                    places=place)
    num_epoch = 2
    for epoch in range(1, num_epoch+1):
        print(f'Epoch [{epoch}/{num_epoch}]:')
        for idx, (data, label) in enumerate(dataloader):
            print(f'Iter {idx}, Data shape:{data.shape}, Label shape:{label.shape}')
    '''
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
