import os
import cv2


if __name__ == "__main__":

    data_dir = r"../dataset/dummy_data/GroundTruth_trainval_png/"

    classes = set()

    for i, fname in enumerate(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        im = cv2.imread(fpath, flags=cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"Warning {fpath} is None.")
        classes = set(list(classes) + list(im.flatten().tolist()))

    print(f"num_image is {i+i}, classes is {classes}.")
