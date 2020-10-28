import os
import numpy as np
import random
import cv2
import paddle.fluid as fluid
from functools import partial


def resize_transform(data, label, size=(256, 256)):
    data = cv2.resize(data, size, interpolation=cv2.INTER_LINEAR)
    """
        因为分割图的值和label有对应关系，而线性插值有概率会产生新值, 所以采用最近邻.
    """
    label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)  # trick.
    return data, label


class BasicDataLoader():
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):

        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle
        self.record_list = []
        self.read_list()

    def read_list(self):
        with open(self.image_list_file) as fr:
            for line in fr.readlines():
                data_rel_path, label_rel_path = line.strip().split()
                data_abs_path, label_abs_path = \
                    self.image_folder + os.sep + data_rel_path, self.image_folder + os.sep + label_rel_path
                assert os.path.exists(
                    data_abs_path) and os.path.exists(label_abs_path)
                self.record_list.append((data_abs_path, label_abs_path))
        # todo:
        if self.shuffle:
            random.shuffle(self.record_list)

    def __len__(self):
        return len(self.record_list)

    def preprocess(self, data, label):
        assert np.array(data.shape[:2] == label.shape[:2]).all()
        if self.transform:
            data, label = self.transform(data, label)
        label = label[..., np.newaxis]  # add channel axis.
        return data, label

    def __call__(self):
        for data, label in self.record_list:
            hwc_im = cv2.imread(data, cv2.IMREAD_COLOR)[..., ::-1]
            gray_im = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            yield self.preprocess(hwc_im, gray_im)


if __name__ == "__main__":
    batch_size = 5
    place = fluid.CPUPlace()
    print("Use CPU")

    with fluid.dygraph.guard(place):
        # craete BasicDataloder instance
        image_folder = "./dummy_data"
        image_list_file = "./dummy_data/list.txt"
        resize_256 = partial(resize_transform, size=(512, 512))
        data_gen = BasicDataLoader(image_folder,
                                   image_list_file,
                                   resize_256,
                                   True)
        # craete fluid.io.DataLoader instance
        dataloader = fluid.io.DataLoader.from_generator(
            capacity=1, use_multiprocess=False)
        # set sample generator for fluid dataloader
        dataloader.set_sample_generator(data_gen, batch_size,
                                        drop_last=False, places=place)

        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(dataloader):
                print(
                    f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')
