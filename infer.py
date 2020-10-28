import os
import cv2
import numpy as np
from PIL import Image
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import to_variable
from modules.pspnet import PSPNet
from modules.transforms import *


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def save_blend_image(image_file, pred_file, dump_dir=""):
    image1 = Image.open(image_file)
    image2 = Image.open(pred_file)
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')
    image = Image.blend(image1, image2, 0.5)
    o_file = pred_file[0:-4] + "_blend.png"
    image.save(o_file)


def inference_resize(hwc_im, size=256):
    h, w = hwc_im.shape[:2]
    scale_x, scale_y = size / w, size / h
    out = cv2.resize(hwc_im, None, fx=scale_x, fy=scale_y,
                     interpolation=cv2.INTER_LINEAR)
    return out, scale_x, scale_y


def inference_deresize(hwc_im, fx, fy):
    return cv2.resize(hwc_im, None, fx=1./fx, fy=1./fy, interpolation=cv2.INTER_NEAREST)


def inference_sliding(hwc_im, kernel_size, stride=1, padding="same"):
    """Not Impl"""
    assert padding in ("same", "valid")
    H, W = hwc_im.shape[:2]

    (H - kernel_size) / stride

    for h in range(0, (H - kernel_size) // stride, stride):
        for w in range(0, W - kernel_size, stride):
            yield hwc_im[h * kernel_size: (h+1) * kernel_size, w * kernel_size: (w+1) * kernel_size]


def inference_multi_scale():
    pass


def save_images(out, out_save_path):
    cv2.imwrite(out_save_path, out)


# this inference code reads a list of image path, and do prediction for each image one by one
def main():
    # 0. env preparation
    with dygraph.guard():
        # 1. create model
        model = PSPNet(3, 60)
        model.eval()
        # 2. load pretrained model
        pretrain_model_path = "./output/PSPNet/Epoch-8-Loss-2.7353980255126955.pdparams"
        model.load_dict(dygraph.load_dygraph(pretrain_model_path)[0])
        # 3. read test image list
        test_dir = r"./dataset/dummy_data/JPEGImages"
        image_list = [os.path.join(test_dir, fname)
                      for fname in os.listdir(test_dir)]
        # 4. create transforms for test image, transform should be same as training
        size, ratio = 256, 0.875
        mean, std = 0, 1
        transform = Compose(
            Resize(size / ratio),
            CenterCrop(size),
            Normalize(mean, std),
            ConvertDataType(),
            HWC2CHW()
        )
        dump_dir = "test_output"
        os.makedirs(dump_dir, exist_ok=True)
        # 5. loop over list of images
        for image_path in image_list:
            # 6. read image and do preprocessing
            assert os.path.exists(image_path), f"{image_path} not exists."
            cv_img = cv2.imread(image_path)[..., ::-1]
            # Resize
            resized_img, fx, fy = inference_resize(cv_img, 256)
            transformed_img, label = transform(resized_img)
            # 7. image to variable
            inp = to_variable(transformed_img[np.newaxis, ...])
            # 8. call inference func
            pred = model(inp)
            pred = fluid.layers.argmax(pred, axis=1)
            pred = pred[0].numpy().astype("uint8")
            pred = inference_deresize(pred, fx, fy)

            # 9. save results
            image_name = os.path.basename(image_path)
            name, suffix = image_name.split(".")
            out_path = os.path.join(dump_dir, f"{name}_pred.{suffix}")
            save_images(pred, out_path)
            save_blend_image(image_path, out_path, dump_dir)


if __name__ == "__main__":
    main()
