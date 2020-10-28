import numpy as np
from typing import List
from functools import reduce

__all__ = ["acc", "mpa", "miou"]


def acc(predict: np.ndarray, target: np.ndarray):
    """"""
    return (predict == target).sum() / reduce(lambda x, y: x * y, target.shape)


def pa(predict_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-8):
    """TP / (TP + FN)"""
    return np.logical_and(predict_mask, target_mask).sum() / (target_mask.sum() + eps)


def iou(predict_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-8):
    """TP / (TP + FP + FN)"""
    return np.logical_and(predict_mask, target_mask).sum() / (np.logical_or(predict_mask, target_mask).sum() + eps)


def mpa(predict: np.ndarray, target: np.ndarray, classes: List[int]):
    assert predict.shape == target.shape, f"{predict.shape} vs {target.shape}"
    pas = []
    for class_i in classes:
        target_mask = target == class_i
        predict_mask = predict == class_i
        pas.append(pa(predict_mask, target_mask))
    return pas


def miou(predict: np.ndarray, target: np.ndarray, classes: List[int]):
    # print(predict)
    # print(target)
    assert predict.shape == target.shape, f"{predict.shape} vs {target.shape}"
    ious = []
    for class_i in classes:
        target_mask = target == class_i
        predict_mask = predict == class_i
        ious.append(iou(predict_mask, target_mask))

    return ious


if __name__ == "__main__":

    np.random.seed(2020)

    xmin, ymin = 4, 5
    xmax, ymax = 12, 12

    n, c, h, w = 1, 8, 20, 20
    ignore_class = 0
    true_class = 6
    classes = range(c)

    pred = np.random.normal(size=(n, c, h, w))
    pred_max = np.max(pred)

    target = np.random.randint(ignore_class+1, c, size=(n, h, w))

    miou_v = miou(pred.argmax(1), target, classes)
    mpa_v = mpa(pred.argmax(1), target, classes)
    acc_v = acc(pred.argmax(1), target)

    print(
        f"""\racc: {acc_v}
            \rmpa: {mpa_v[true_class]}
            \rmiou: {miou_v[true_class]}
            """)

    pred1 = pred.copy()
    # pred1[..., nd_range] = true_class
    pred1[..., true_class, ymin: ymax, xmin: xmax] = pred_max + 0.1
    target1 = target.copy()
    target1[..., ymin: ymax, xmin: xmax] = true_class

    miou_v = miou(pred1.argmax(1), target1, classes)
    mpa_v = mpa(pred1.argmax(1), target1, classes)
    acc_v = acc(pred1.argmax(1), target1)

    print(
        f"""\racc: {acc_v}
            \rmpa: {mpa_v[true_class]}
            \rmiou: {miou_v[true_class]}
            """)

    pred1 = pred.copy()
    pred1[..., true_class,  ymin: ymax, xmin: xmax] = pred_max + 0.1
    target1 = target.copy()
    target1[...,  ymin: ymax, xmin: xmax] = c - true_class

    miou_v = miou(pred1.argmax(1), target1, classes)
    mpa_v = mpa(pred1.argmax(1), target1, classes)
    acc_v = acc(pred1.argmax(1), target1)

    print(
        f"""\racc: {acc_v}
            \rmpa: {mpa_v[true_class]}
            \rmiou: {miou_v[true_class]}
            """)
