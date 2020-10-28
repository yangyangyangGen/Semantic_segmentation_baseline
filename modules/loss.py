import paddle.fluid as fluid

__all__ = ["cross_entropy"]


def cross_entropy(preds: fluid.Variable,
                  labels: fluid.Variable,
                  ignore_index: int = 255,
                  eps: float = 1e-8):
    """
    @param preds: [N, C, H, W]
    @param labels: [N, H, W, 1]
    @param ignore_index: 
    @param eps
    """
    assert preds.shape[2:] == labels.shape[1:3], \
        f"{preds.shape} vs {labels.shape}"
    assert len(labels.shape) == 4

    preds = fluid.layers.cast(preds, "float32")
    labels = fluid.layers.cast(labels, "int64")

    # [n, c, h, w] -> [n, h, w, c]
    preds_hwc = fluid.layers.transpose(preds, [0, 2, 3, 1])

    mask = labels != ignore_index
    mask = fluid.layers.cast(mask, 'float32')

    # call criterion and compute loss
    loss = criterion = fluid.layers.softmax_with_cross_entropy(
        preds_hwc, labels, ignore_index=ignore_index)

    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + eps)
    return avg_loss


if __name__ == "__main__":
    import os
    import numpy as np
    import cv2

    cwd = os.path.dirname(os.path.abspath(__file__))
    label = cv2.imread(cwd + os.sep +
                       '../dataset/dummy_data/GroundTruth_trainval_png/2008_000203.png')
    assert label is not None
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.int64)

    pred = np.random.uniform(
        0, 1, (1, 59, label.shape[0], label.shape[1])).astype(np.float32)
    label = label[np.newaxis, ..., np.newaxis]

    with fluid.dygraph.guard():
        pred = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss = cross_entropy(pred, label)
        print(loss)

    def test():

        N, C, H, W = 1, 59, 256, 256
        target_label = 10
        fake_label = 10

        """
            loss_j =  -\\text{logits}_{label_j} +
            \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{logits}_i)\\right), j = 1,..., K
        """
        label = np.ones(shape=(N, H, W, 1)).astype("int64") * target_label
        pred = np.zeros(shape=(N, C, H, W), dtype="float32")
        pred[:, fake_label, ...] = 10

        pred2 = np.zeros(shape=(N, C, H, W), dtype="float32")

        with fluid.dygraph.guard():
            pred = fluid.dygraph.to_variable(pred)
            label = fluid.dygraph.to_variable(label)
            loss = cross_entropy(pred, label)
            print(loss)

            pred2 = fluid.dygraph.to_variable(pred2)
            loss2 = cross_entropy(pred2, label)
            print(loss2)

    print("Start test")
    test()
    print("End test")
