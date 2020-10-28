import cv2
import numpy as np
import random


class Compose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1/255.0 if val_scale == 1 else 1

    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label


class HWC2CHW(object):
    def __call__(self, hwc_image, label=None):
        # [H, W, C] -> [C, H, W]
        hwc_image = np.transpose(hwc_image, (2, 0, 1))
        return hwc_image, label


class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255)
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val = mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple, list)):
            self.mean_val = [int(x * factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)

    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = int(max(self.size - h, 0))
        pad_w = int(max(self.size - w, 0))

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


class CenterCrop(object):
    def __init__(self, hw: int = 256):
        super(CenterCrop, self).__init__()

        if isinstance(hw, (tuple, list)):
            assert len(hw) == 2
            self.hw = tuple(map(int, hw))
        else:
            self.hw = (int(hw), int(hw))

    def __call__(self, cv_img: np.ndarray, label: np.ndarray = None):
        h, w = cv_img.shape[:2]
        ch, cw = h / 2., w / 2.
        crop_h, crop_w = self.hw[0] / 2., self.hw[1] / 2.

        crop_xmin, crop_ymin = max(int(cw - crop_w), 0), \
            max(int(ch - crop_h), 0)
        crop_xmax, crop_ymax = min(int(cw + crop_w), w), \
            min(int(cw + crop_h), h)

        crop_img = cv_img[crop_ymin: crop_ymax, crop_xmin: crop_ymax, :]
        if label is None:
            return crop_img, label

        crop_label = label[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        # resize or pad ???
        out_img = cv2.resize(crop_img, tuple(self.hw),
                             interpolation=cv2.INTER_LINEAR)
        out_label = cv2.resize(crop_label, tuple(
            self.hw), interpolation=cv2.INTER_NEAREST)

        return (out_img, out_label)


class Resize(object):
    def __init__(self, hw):
        super(Resize, self).__init__()
        if isinstance(hw, (tuple, list)):
            assert len(hw) == 2
            self.hw = tuple(map(int, hw))
        else:
            self.hw = (int(hw), int(hw))

    def __call__(self, image, label=None):
        image = cv2.resize(image, self.hw,
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, self.hw,
                               interpolation=cv2.INTER_NEAREST)
        return image, label


class RandomInterpolationsResize(object):
    def __init__(self, hw,
                 interpolations=(cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                 cv2.INTER_AREA, cv2.INTER_CUBIC,
                                 cv2.INTER_LANCZOS4)):
        super(RandomInterpolationsResize, self).__init__()

        assert len(interpolations)

        self.interpolations = interpolations
        if isinstance(hw, (tuple, list)):
            assert len(hw) == 2
            self.hw = tuple(map(int, hw))
        else:
            self.hw = (int(hw), int(hw))

    def __call__(self, image, label=None):
        image = cv2.resize(image, self.hw,
                           interpolation=random.choice(self.interpolations))
        if label is not None:
            label = cv2.resize(label, self.hw,
                               interpolation=cv2.INTER_NEAREST)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, ratio=.5):
        super(RandomHorizontalFlip, self).__init__()
        assert 0 < ratio < 1, f"`ratio`: {ratio} should be > 0 and < 1."
        self._ratio = ratio

    def __call__(self, hwc_image, label):
        # uniform.
        if random.random() > self._ratio:
            hwc_image = hwc_image[:, ::-1, :]
            label = label[:, ::-1]
        return hwc_image, label


class RandomVerticalFlip(object):
    def __init__(self, ratio=.5):
        super(RandomVerticalFlip, self).__init__()
        assert 0 < ratio < 1, f"`ratio`: {ratio} should be > 0 and < 1."
        self._ratio = ratio

    def __call__(self, hwc_image, label):
        if random.random() > self._ratio:
            hwc_image = hwc_image[::-1, :, :]
            label = label[::-1, :]
        return hwc_image, label


class RandomCropResize(object):
    def __init__(self, hw=256):
        super(RandomCrop, self).__init__()

        if isinstance(hw, (tuple, list)):
            assert len(hw) == 2
            self.hw = tuple(map(int, hw))
        else:
            self.hw = (int(hw), int(hw))

    def __call__(self, hwc_img, label):
        h, w = hwc_img.shape[:2]

        # -1 : because random.randint(a, b) return [a, b].
        crop_xmin = random.randint(0, max(w - self.hw[1] - 1, 0))
        crop_ymin = random.randint(0, max(h - self.hw[0] - 1, 0))

        crop_xmax, crop_ymax = min(crop_xmin + self.hw[1], w), \
            min(crop_ymin + self.hw[0], h)

        assert crop_xmax >= crop_xmin or crop_ymax >= crop_ymin, f"assert {crop_ymin} {crop_ymax} {crop_xmin} {crop_xmax} {h} {w}"

        crop_img = hwc_img[crop_ymin: crop_ymax, crop_xmin: crop_xmax, :]
        crop_label = label[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        crop_hw = crop_img.shape[:2]
        if crop_hw[0] < self.hw[0] or crop_hw[1] < self.hw[1]:
            # resize or padding.
            crop_img = cv2.resize(crop_img, tuple(self.hw),
                                  interpolation=cv2.INTER_LINEAR)
            crop_label = cv2.resize(crop_label, tuple(
                self.hw), interpolation=cv2.INTER_NEAREST)

        return (crop_img, crop_label)


class Scale(object):
    def __init__(self, xy_scale,
                 interpolations=(cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                 cv2.INTER_AREA, cv2.INTER_CUBIC,
                                 cv2.INTER_LANCZOS4)):
        super(Scale, self).__init__()

        assert len(interpolations)
        self.interpolations = interpolations

        if isinstance(xy_scale, (tuple, list)):
            assert len(xy_scale) == 2
            self.xy_scale_tuple = tuple(xy_scale)
        else:
            self.xy_scale_tuple = (float(xy_scale), float(xy_scale))

    def __call__(self, cv_image, label=None):
        x_scale, y_scale = self.xy_scale_tuple
        cv_image = cv2.resize(cv_image, dsize=None, fx=x_scale, fy=y_scale,
                              interpolation=random.choice(self.interpolations))

        if label is None:
            return cv_image

        label = cv2.resize(cv_image, dsize=None, fx=x_scale, fy=y_scale,
                           interpolation=cv2.INTER_NEAREST)

        return cv_image, label


class RandomScale(object):
    def __init__(self, xy_scale_list,
                 interpolations=(cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                 cv2.INTER_AREA, cv2.INTER_CUBIC,
                                 cv2.INTER_LANCZOS4)):
        super(RandomScale, self).__init__()

        assert len(interpolations)
        assert isinstance(xy_scale_list, (tuple, list))
        self.interpolations = interpolations
        self.xy_scale_list = tuple(xy_scale_list)

    def __call__(self, cv_image, label=None):
        scale = random.choice(self.xy_scale_list)
        if not isinstance(scale, (tuple, list)):
            scale = (scale, ) * 2
        x_scale, y_scale = scale

        cv_image = cv2.resize(cv_image, dsize=None,
                              fx=x_scale, fy=y_scale,
                              interpolation=random.choice(self.interpolations))
        if label is not None:
            label = cv2.resize(label, dsize=None,
                               fx=x_scale, fy=y_scale,
                               interpolation=cv2.INTER_NEAREST)

        return cv_image, label


if __name__ == "__main__":
    import os
    cwd = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(
        cwd + os.sep + 'dummy_data/JPEGImages/2008_000064.jpg')
    label = cv2.imread(
        cwd + os.sep + 'dummy_data/GroundTruth_trainval_png/2008_000064.png')
    assert image is not None and label is not None

    crop_size = 256
    transform = Compose([
        # RandomScale(xy_scale_list=[(.5, .5), (1., 1.), (2., 2.)]),
        RandomCropResize(hw=crop_size / 0.875),
        CenterCrop(hw=crop_size)
    ])

    save_path = "test_transforms_output"
    os.makedirs(save_path, exist_ok=True)
    for i in range(10):
        out_image, out_label = transform(image, label)
        # save image.
        if 0:
            cv2.imwrite(save_path + os.sep + f'{i}.jpg', out_image)
            cv2.imwrite(save_path + os.sep + f'{i}.png', out_label)
