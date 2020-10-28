import numpy as np

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

    def update(self, ncls: int, ignore_label: int):
        assert ncls >= 1 and ignore_label >= 0
        self.ncls = ncls
        self.ignore_label = ignore_label

    def __call__(self, predict: np.ndarray, target: np.ndarray):
        """
            predict: [N, C, H, W]
            target: [N, H, W]
        """

        # 1. check
        assert predict is not None and target is not None
        assert predict.ndim == 4 and target.ndim == 3, \
            f"predict ndim {predict.shape} vs 4, and target ndim {target.shape} vs 3."
        predict = predict.argmax(1)
        assert predict.shape == target.shape, f"{predict.shape} vs {target.shape}"

        # 2. get score matrix.
        predict = predict.astype(np.int64)
        target = target.astype(np.int64)

        k = (target >= self.ignore_label) & (target < self.ncls)
        # k = np.logical_and(target >= self.ignore_label, target < self.ncls)

        score_matrix = np.bincount(self.ncls * target[k] + predict[k],
                                   minlength=self.ncls ** 2).reshape(
                                       (self.ncls, ) * 2)

        # 3. get metric.
        tp = np.diag(score_matrix)
        acc = tp.sum() / (score_matrix.sum() + self.eps)
        mpa = tp / (score_matrix.sum(1) + self.eps)
        miou = tp / ((score_matrix.sum(1) +
                      score_matrix.sum(0) - tp) + self.eps)

        return {"acc": acc,
                "mpa": mpa,
                "miou": miou}


if __name__ == "__main__":

    np.random.seed(2020)

    # pred_range = np.array(((130, 150), (180, 220)))
    # target_range = np.array(((130, 150), (180, 220)))
    # nd_range = np.array(((130, 150), (180, 220)))
    xmin, ymin = 4, 5
    xmax, ymax = 12, 12

    n, c, h, w = 5, 8, 20, 20
    ignore_class = 0
    true_class = 6

    pred = np.random.normal(size=(n, c, h, w))
    pred_max = np.max(pred)

    target = np.random.randint(
        ignore_class+1, c, size=(n, h, w), dtype=np.int64)

    seg_metrics = SegmentationMetrics(c, ignore_label=ignore_class)
    info = seg_metrics(pred, target)

    print(f"""\racc: {info['acc']} 
            \rmpa: {info['mpa'][true_class]}  
            \rmiou: {info['miou'][true_class]}
            \rreal miou: {sum(info['miou'])/len(info['miou'])}
            """)

    pred1 = pred.copy()
    # pred1[..., nd_range] = true_class
    pred1[..., true_class, ymin: ymax, xmin: xmax] = pred_max + 0.1
    target1 = target.copy()
    target1[..., ymin: ymax, xmin: xmax] = true_class

    info = seg_metrics(pred1, target1)
    print(f"""\racc: {info['acc']} 
            \rmpa: {info['mpa'][true_class]}  
            \rmiou: {info['miou'][true_class]}
            \rreal miou: {sum(info['miou'])/len(info['miou'])}
            """)

    pred1 = pred.copy()
    pred1[..., true_class,  ymin: ymax, xmin: xmax] = pred_max + 0.1
    target1 = target.copy()
    target1[...,  ymin: ymax, xmin: xmax] = c - true_class

    info = seg_metrics(pred1, target1)
    print(f"""\racc: {info['acc']} 
            \rmpa: {info['mpa'][true_class]}  
            \rmiou: {info['miou'][true_class]}
            \rreal miou: {sum(info['miou'])/len(info['miou'])}
            """)
