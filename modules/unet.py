import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Sequential


class Encoder(Layer):
    def __init__(self, num_channels, num_filters,
                 filter_size=3, stride=1, padding=0,
                 act="relu", pool_type="max"):
        super(Encoder, self).__init__()
        # TODO: encoder contains:
        #       1 3x3conv + 1bn + relu +
        #       1 3x3conc + 1bn + relu +
        #       1 2x2 pool
        # return features before and after pool

        self.conv1 = Sequential(
            Conv2D(num_channels, num_filters,
                   filter_size=filter_size, stride=stride, padding=padding),
            BatchNorm(num_filters, act=act)
        )
        self.conv2 = Sequential(
            Conv2D(num_filters, num_filters,
                   filter_size=filter_size, stride=stride, padding=padding),
            BatchNorm(num_filters, act=act)
        )
        self.pool = Pool2D(pool_size=2, pool_stride=2, pool_type=pool_type)

    def forward(self, inp):
        # TODO: finish inference part
        x = self.conv1(inp)
        x = self.conv2(x)
        x_pooled = self.pool(x)

        return x, x_pooled


class Decoder(Layer):
    def __init__(self, num_channels, num_filters,
                 filter_size=2, stride=2, padding=0, act="relu"):
        super(Decoder, self).__init__()
        # TODO: decoder contains:
        #       1 2x2 transpose conv (makes feature map 2x larger)
        #       1 3x3 conv + 1bn + 1relu +
        #       1 3x3 conv + 1bn + 1relu

        self.trans_conv = Sequential(
            Conv2DTranspose(num_channels, num_filters,
                            filter_size=filter_size,
                            stride=stride, padding=padding),
            BatchNorm(num_filters, act=act)
        )

        self.conv1 = Sequential(
            Conv2D(num_filters * 2, num_filters, filter_size=3),
            BatchNorm(num_filters, act=act)
        )

        self.conv2 = Sequential(
            Conv2D(num_filters, num_filters, filter_size=3),
            BatchNorm(num_filters, act=act)
        )

    def forward(self, inp_prev, inp):
        # forward contains an Pad2d and Concat
        x = self.trans_conv(inp)
        # center crop and concat.
        h, w = x.shape[2:]
        prev_h, prev_w = inp_prev.shape[2:]
        assert prev_h > h and prev_w > w

        prev_ch, prev_cw = prev_h // 2, prev_w // 2
        xmin, ymin = prev_cw - w // 2, prev_ch - h // 2
        xmax, ymax = xmin + w, ymin + h

        assert xmax <= prev_w and ymax <= prev_h

        crop = inp_prev[:, :, ymin: ymax, xmin: xmax]

        x = fluid.layers.concat([crop, x], axis=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(Layer):
    def __init__(self, num_inp=3, num_classes=59,
                 filter_size=3, stride=1, padding=0, act="relu", pool_type="max",
                 up_filter_size=2, up_stride=2, up_padding=0, up_act="relu"):
        super(UNet, self).__init__()
        # 4 encoders, 4 decoders, and mid layers contains 2 1x1conv+bn+relu

        # encoder: channel: `num_inp`->64->128->256->512
        self.down1 = Encoder(num_inp, 64,
                             filter_size=filter_size, stride=stride, padding=padding, act=act, pool_type=pool_type)
        self.down2 = Encoder(64, 128,
                             filter_size=filter_size, stride=stride, padding=padding, act=act, pool_type=pool_type)
        self.down3 = Encoder(128, 256,
                             filter_size=filter_size, stride=stride, padding=padding, act=act, pool_type=pool_type)
        self.down4 = Encoder(256, 512,
                             filter_size=filter_size, stride=stride, padding=padding, act=act, pool_type=pool_type)
        # mid: 512->1024->1024
        self.mid_conv1 = Conv2D(512, 1024,
                                filter_size=filter_size, stride=stride, padding=padding, act=None)
        self.mid_bn1 = BatchNorm(1024, act=act)

        self.mid_conv2 = Conv2D(1024, 1024,
                                filter_size=filter_size, stride=stride, padding=padding, act=None)
        self.mid_bn2 = BatchNorm(1024, act=act)

        # up: 1024->512->256->128->64
        self.up1 = Decoder(1024, 512,
                           filter_size=up_filter_size, stride=up_stride,
                           padding=up_padding, act=act)

        self.up2 = Decoder(512, 256,
                           filter_size=up_filter_size, stride=up_stride,
                           padding=up_padding, act=act)

        self.up3 = Decoder(256, 128,
                           filter_size=up_filter_size, stride=up_stride,
                           padding=up_padding, act=act)

        self.up4 = Decoder(128, 64,
                           filter_size=up_filter_size, stride=up_stride,
                           padding=up_padding, act=act)
        # header: 64 -> num_classes
        self.header = Sequential(
            Conv2D(64, num_classes, filter_size=1)
        )

    def forward(self, inp):
        x1, x = self.down1(inp)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.header(x)
        x = fluid.layers.interpolate(x, inp.shape[2:])

        return x


if __name__ == "__main__":

    with fluid.dygraph.guard():
        model = UNet(num_classes=59)
        # x_data = np.random.rand(1, 3, 123, 123).astype(np.float32)
        x_data = np.random.rand(1, 3, 572, 572).astype(np.float32)
        inputs = to_variable(x_data)
        pred = model(inputs)
        print(inputs.shape)
        print(pred.shape)
