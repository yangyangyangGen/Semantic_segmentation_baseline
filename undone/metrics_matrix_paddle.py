import numpy as np
from paddle import fluid
from paddle.fluid.dygraph import to_variable

__all__ = ["SegmentationMetrics"]


class SegmentationMetrics(object):
    def __init__(self,
                 ncls: int = 0,
                 ignore_label: int = 0,
                 eps: float = 1e-8):
        """
        See https://arxiv.org/pdf/1704.06857.pdf.
        @param
            ncls: Range must be continuous.
        """
        self.eps = eps
        self.ncls = ncls
        self.ignore_label = ignore_label

    def update(self, ncls, ignore_label):
        assert ncls >= 1 and ignore_label >= 0
        self.ncls = ncls
        self.ignore_label = ignore_label

    def __call__(self, predict: fluid.Variable, target: fluid.Variable):
        """
            predict: [N, C, H, W]
            target: [N, H, W]
        """

        # 1. check
        assert predict is not None and target is not None
        predict = fluid.layers.argmax(predict, axis=1)

        miou, out_wrong, out_correct = fluid.layers.mean_iou(
            predict, target, self.ncls)


        return miou, out_wrong, out_correct


if __name__ == "__main__":

    np.random.seed(2020)

    xmin, ymin = 4, 5
    xmax, ymax = 12, 12

    n, c, h, w = 5, 8, 20, 20
    ignore_class = 0
    true_class = 6

    pred = np.random.normal(size=(n, c, h, w))
    pred_max = np.max(pred)

    target = np.random.randint(
        ignore_class+1, c, size=(n, h, w), dtype=np.int64)

    pred1 = pred.copy()
    # pred1[..., nd_range] = true_class
    pred1[..., true_class, ymin: ymax, xmin: xmax] = pred_max + 0.1
    target1 = target.copy()
    target1[..., ymin: ymax, xmin: xmax] = true_class

    pred2 = pred.copy()
    pred2[..., true_class,  ymin: ymax, xmin: xmax] = pred_max + 0.1
    target2 = target.copy()
    target2[...,  ymin: ymax, xmin: xmax] = c - true_class

    seg_metrics = SegmentationMetrics(c)

    with fluid.dygraph.guard():
        pred = to_variable(pred)
        target = to_variable(target)
        info = seg_metrics(pred, target)
        print(*info)

        pred = to_variable(pred1)
        target = to_variable(target1)
        info = seg_metrics(pred, target)
        print(*info)

        pred = to_variable(pred2)
        target = to_variable(target2)
        info = seg_metrics(pred, target)
        print(*info)
