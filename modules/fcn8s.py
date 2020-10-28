import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Sequential
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from .backbone_models.vgg import VGG16BN


class FCN8s(fluid.dygraph.Layer):
    # create fcn8s model

    #  TransposeConvolution params.
    def __init__(self,
                 num_inp, num_out,
                 act="relu",
                 backbone_use_pretrain=False):
        super(FCN8s, self).__init__()
        backbone = VGG16BN(backbone_use_pretrain)

        self.layer1 = backbone.layer1
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type="max")
        self.layer2 = backbone.layer2
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type="max")
        self.layer3 = backbone.layer3
        self.pool3 = Pool2D(pool_size=2, pool_stride=2, pool_type="max")
        self.layer4 = backbone.layer4
        self.pool4 = Pool2D(pool_size=2, pool_stride=2, pool_type="max")
        self.layer5 = backbone.layer5
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, pool_type="max")

        self.pool3_score = Sequential(
            Conv2D(num_channels=256, num_filters=num_out, filter_size=1,
                   act=None),
            BatchNorm(num_channels=num_out, act=act)
        )
        self.pool4_score = Sequential(
            Conv2D(num_channels=512, num_filters=num_out, filter_size=1,
                   act=None),
            BatchNorm(num_channels=num_out, act=act)
        )
        self.pool5_score = Sequential(
            Conv2D(num_channels=512, num_filters=num_out, filter_size=1,
                   act=None),
            BatchNorm(num_channels=num_out, act=act)
        )

        """
            论文中这里的Conv2DTranspose使用非标准倍采样后再进行crop，
            这里没有做.
            
            论文crop的 offset_start
            5  for pool5_score
            9  for pool4_score
            31 for pool3_score
        """
        self.pool_upsample1 = Sequential(
            Conv2DTranspose(num_channels=num_out, num_filters=num_out,
                            filter_size=2, padding=0, stride=2, act=None),
            BatchNorm(num_out, act=act)
        )

        self.pool_upsample2 = Sequential(
            Conv2DTranspose(num_channels=num_out, num_filters=num_out,
                            filter_size=2, padding=0, stride=2, act=None),
            BatchNorm(num_out, act=act)
        )

        self.header = Sequential(
            Conv2DTranspose(num_channels=num_out, num_filters=num_out,
                            filter_size=10, padding=1, stride=8, act=None),
            BatchNorm(num_channels=num_out, act=act)
        )

    def forward(self, inp):
        x = self.layer1(inp)  # /2
        x = self.pool1(x)
        x = self.layer2(x)  # /4
        x = self.pool2(x)
        x = self.layer3(x)  # /8
        x = self.pool3(x)
        pool3 = x
        x = self.layer4(x)  # /16
        x = self.pool4(x)
        pool4 = x
        x = self.layer5(x)  # /32
        x = self.pool5(x)

        #
        x = self.pool5_score(x)
        x = self.pool_upsample1(x)
        pool4_score = self.pool4_score(pool4)
        x = pool4_score + x

        x = self.pool_upsample2(x)
        pool3_score = self.pool3_score(pool3)
        x = pool3_score + x

        x = self.header(x)

        return x


if __name__ == '__main__':
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = FCN8s(3, 59)
        model.eval()
        pred = model(x)
        print(f"input shape is {x_data.shape}")
        print(f"output shape is {pred.shape}")
