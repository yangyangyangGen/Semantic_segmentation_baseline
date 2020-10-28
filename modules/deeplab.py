import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import Sequential
from .backbone_models.resnet_multi_grid import ResNet50
from functools import partial

__all__ = ["DeepLabV3", "DeepLabV3Plus"]


class ASPPPooling(Layer):
    def __init__(self, num_in, num_out, ap_out_size=(1, 1)):
        super(ASPPPooling, self).__init__()
        assert type(ap_out_size) in (tuple, list)
        self.ap_out_size = ap_out_size
        self.conv = Sequential(
            Conv2D(num_in, num_out, filter_size=1),
            BatchNorm(num_out, act="relu")
        )

    def forward(self, inp):
        x = fluid.layers.adaptive_pool2d(inp, pool_size=self.ap_out_size)
        x = self.conv(x)
        x = fluid.layers.interpolate(x, out_shape=inp.shape[2:])
        return x


class ASPPConv(Sequential):
    def __init__(self, num_in, num_out, dilation, stride=1, act="relu"):
        # 2 * padding must equal (kernel_size - 1) * dilation
        super(ASPPConv, self).__init__(
            Conv2D(num_in, num_out, filter_size=3,
                   stride=stride, padding=dilation, dilation=dilation),
            BatchNorm(num_out, act=act)
        )


class ASPPModule(Layer):
    def __init__(self, num_in, num_out, num_dilates):
        super(ASPPModule, self).__init__()
        self.modules = []
        self.modules.append(
            Sequential(
                Conv2D(num_in, num_out, filter_size=1),
                BatchNorm(num_out, act="relu")
            )
        )
        for dilation in num_dilates:
            self.modules.append(
                ASPPConv(num_in, num_out, dilation)
            )
        self.modules.append(ASPPPooling(num_in, num_out))

        self.fusing_conv = Sequential(
            Conv2D(len(self.modules) * num_out, num_out, filter_size=1),
            BatchNorm(num_out, act="relu")
        )

    def forward(self, inp):
        """
        i = 0
        for module in self.modules:
            out = module(inp)
            print(f"{i} -> {out.shape}")
            i += 1
        exit(1)
        """
        x = fluid.layers.concat([module(inp)
                                 for module in self.modules], axis=1)
        return self.fusing_conv(x)


class DeepLabHead(fluid.dygraph.Sequential):
    def __init__(self, num_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPPModule(num_channels, 256, [12, 24, 36]),
            Conv2D(256, 256, 3, padding=1),
            BatchNorm(256, act='relu'),
            Conv2D(256, num_classes, 1)
        )


class DeepLabV3(Layer):
    def __init__(self, num_inp, num_classes,
                 backbone="resnet50", backbone_pretrained=False):
        super(DeepLabV3, self).__init__()

        support_backbone = ("resnet50")
        assert backbone in support_backbone, f"sorry not support{backbone} now!"

        backbone = ResNet50(pretrained=backbone_pretrained,
                            duplicate_blocks=True)

        self.stem = Sequential(
            backbone.conv,
            backbone.pool2d_max
        )

        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        self.stage5 = backbone.layer5
        self.stage6 = backbone.layer6
        self.stage7 = backbone.layer7

        self.head = DeepLabHead(2048, num_classes)

    def forward(self, inp):
        x = self.stem(inp)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.head(x)
        x = fluid.layers.interpolate(x, inp.shape[2:])
        return x


class DeepLabV3Plus(Layer):
    # TODO:
    pass


def main():
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = DeepLabV3(num_classes=59)
        model.eval()
        pred = model(x)
        print(f"pred shape {pred.shape}")


if __name__ == '__main__':
    main()
