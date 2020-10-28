import numpy as np
import paddle.fluid as fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Sequential
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout

from .backbone_models.resnet_dilated import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from functools import partial

# pool with different bin_size
# interpolate back to input size
# concat


class PSPModule(dygraph.Layer):
    """
        multi scale Ap Pool and concat.
        output channel equal input_channel.
    """

    def __init__(self, num_channels=None,
                 psp_size_list=(1, 2, 3, 6),
                 act="relu"):
        super(PSPModule, self).__init__()
        assert num_channels is not None
        assert num_channels % len(psp_size_list) == 0
        each_psp_len = num_channels // len(psp_size_list)

        self.modules = []
        for psp_size in psp_size_list:
            assert isinstance(psp_size, int)
            # TODO: add upsample layer.
            self.modules.append((
                partial(fluid.layers.adaptive_pool2d,
                        pool_size=psp_size, pool_type="max"),
                Sequential(
                    Conv2D(num_channels=num_channels, num_filters=each_psp_len,
                           filter_size=1, stride=1, padding=0, act=None),
                    BatchNorm(each_psp_len, act=act))
            ))

    def forward(self, inp):
        out_seq = [fluid.layers.interpolate(
            module[1](module[0](inp)), out_shape=inp.shape[2:]) for module in self.modules]
        return fluid.layers.concat([*out_seq, inp], axis=1)


class PSPNet(Layer):
    def __init__(self, num_inp, num_classes=59,
                 act="relu",
                 backbone='resnet50', backbone_pretrained=False):
        super(PSPNet, self).__init__(name_scope="PSPNet")

        model_dict = {'resnet18':  ResNet18,
                      'resnet34':  ResNet34,
                      'resnet50':  ResNet50,
                      'resnet101': ResNet101,
                      'resnet152': ResNet152}

        assert backbone in model_dict.keys(), \
            f"`backbone` must in {model_dict.keys()}"

        backbone = model_dict[backbone](backbone_pretrained)
        # backbone = ResNet50(backbone_pretrained)
        # stem: res.conv, res.pool2d_max
        self.layer0 = Sequential(
            backbone.conv,
            backbone.pool2d_max
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # psp: 2048 -> 2048*2
        self.psp_module = PSPModule(2048, (1, 2, 3, 6))

        # cls: 2048*2 -> 512 -> num_classes
        self.header = Sequential(
            Conv2D(2048 * 2, 512, filter_size=3, stride=1),
            BatchNorm(512, act=act),
            Dropout(0.1),
            Conv2D(512, num_classes, filter_size=1)
        )

        # aux: 1024 -> 256 -> num_classes

    def forward(self, inp):

        # aux: tmp_x = layer3
        x = self.layer0(inp)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.psp_module(x)
        x = self.header(x)
        x = fluid.layers.interpolate(x, inp.shape[2:])
        return x


def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data = np.random.rand(2, 3, 473, 473).astype(np.float32)
        x = to_variable(x_data)
        model = PSPNet(num_classes=59)
        model.train()
        pred, aux = model(x)
        print(pred.shape, aux.shape)


if __name__ == "__main__":
    main()
