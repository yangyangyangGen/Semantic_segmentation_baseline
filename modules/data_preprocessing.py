from basic_transforms import *


class TrainAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1.0):
        # TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops.
        self.augment = Compose([
            RandomScale(xy_scale_list=[
                        (.5, .5), (.8, .8), (1., 1.), (1.5, 1.5), (2., 2.)]),
            Pad(size=image_size/0.8),
            RandomCrop(hw=image_size),
            RandomFlip(),
            ConvertDataType(),
            Normalize(mean_val, std_val),
            HWC2CHW()
        ])

    def __call__(self, image, label):
        return self.augment(image, label)
