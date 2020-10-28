from __future__ import division, print_function, absolute_import

from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import Conv2D, Pool2D, Conv2DTranspose, Sequential, BatchNorm


class Encoder(dygraph.Layer):
    def __init__(self, num_inp,
                 filter_size=3, stride=1, padding=0, act="relu", pool_type="max"):
        super(Encoder, self).__init__()
        self.stage1 = self._get_stage(num_inp, 64,
                                      filter_size=filter_size, stride=stride, padding=padding, act=act)
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type=pool_type)

        self.stage2 = self._get_stage(64, 128,
                                      filter_size=filter_size, stride=stride, padding=padding, act=act)
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type=pool_type)

        self.stage3 = self._get_stage(128, 256,
                                      filter_size=filter_size, stride=stride, padding=padding, act=act)
        self.pool3 = Pool2D(pool_size=2, pool_stride=2, pool_type=pool_type)

        self.stage4 = self._get_stage(256, 512,
                                      filter_size=filter_size, stride=stride, padding=padding, act=act)
        self.pool4 = Pool2D(pool_size=2, pool_stride=2, pool_type=pool_type)

    def _get_stage(self, num_in, num_out, num_block=2,
                   filter_size=3, stride=1, padding=0, act="relu"):
        layers = []
        in_for_block = num_in
        for idx_block in range(num_block):
            layers.append(Sequential(
                Conv2D(in_for_block, num_out,
                       filter_size=filter_size, stride=stride, padding=padding, act=None),
                BatchNorm(num_out, act=act)
            ))
            in_for_block = num_out
        return Sequential(*layers)

    def forward(self, inp):
        stage1 = self.stage1(inp)
        x = self.pool1(stage1)

        stage2 = self.stage2(x)
        x = self.pool2(stage2)

        stage3 = self.stage3(x)
        x = self.pool3(stage3)

        stage4 = self.stage4(x)
        x = self.pool4(stage4)

        return x, (stage1, stage2, stage3, stage4)


class Decoder(dygraph.Layer):
    def __init__(self, filter_size=2, stride=2, padding=0, act="relu"):
        super(Decoder, self).__init__()

        up_sample1 = self._getConv2DTransposeBN(
            1024, 512, filter_size=2, stride=2, padding=0, act=act)
        up_sample1_seq = self._getConv2DBNBlock(
            2, 1024, 512, filter_size=3, stride=1, padding=0, act=act)

        up_sample2 = self._getConv2DTransposeBN(
            512, 256, filter_size=2, stride=2, padding=0, act=act)
        up_sample2_seq = self._getConv2DBNBlock(
            2, 512, 256, filter_size=3, stride=1, padding=0, act=act)

        up_sample3 = self._getConv2DTransposeBN(
            256, 128, filter_size=2, stride=2, padding=0, act=act)
        up_sample3_seq = self._getConv2DBNBlock(
            2, 256, 128, filter_size=3, stride=1, padding=0, act=act)

        up_sample4 = self._getConv2DTransposeBN(
            128, 64, filter_size=2, stride=2, padding=0, act=act)
        up_sample4_seq = self._getConv2DBNBlock(
            2, 128, 64, filter_size=3, stride=1, padding=0, act=act)

        self.up_sample_layers = (
            (up_sample1, up_sample1_seq),
            (up_sample2, up_sample2_seq),
            (up_sample3, up_sample3_seq),
            (up_sample4, up_sample4_seq))

    def _getConv2DTransposeBN(self, num_in, num_out,
                              filter_size=2, stride=2, padding=0, act="relu"):
        return Sequential(
            Conv2DTranspose(num_in, num_out,
                            filter_size=filter_size, stride=stride,
                            padding=padding, act=None),
            BatchNorm(num_out, act=act),
        )

    def _getConv2DBNBlock(self, num_block, num_in, num_out,
                          filter_size=3, stride=1, padding=0, act="relu"):
        layers = []
        in_channel = num_in
        for block in range(num_block):
            layers.append(Sequential(
                Conv2D(in_channel, num_out,
                       filter_size=filter_size, stride=stride,
                       padding=padding, act=None),
                BatchNorm(num_out, act=act),
            ))
            in_channel = num_out
        return Sequential(*layers)

    def forward(self, inp, inp_prevs):
        assert len(inp_prevs) == len(self.up_sample_layers)

        inp_prevs = inp_prevs[::-1]  # reverse.
        x = inp

        for ((up_sample_layer, seq_layer), inp_prev) in \
                zip(self.up_sample_layers, inp_prevs):

            up_sample = up_sample_layer(x)

            up_h, up_w = up_sample.shape[2:]
            prev_h, prev_w = inp_prev.shape[2:]
            assert not (prev_h < up_h or prev_w < up_w)

            # center crop.
            center_h, center_w = prev_h // 2, prev_w // 2
            crop_half_h, crop_half_w = up_h // 2, up_w // 2

            xmin, ymin = center_w - crop_half_w, center_h - crop_half_h
            xmax, ymax = xmin + up_w, ymin + up_h
            prev_crop = inp_prev[:, :, ymin: ymax, xmin: xmax]

            out = fluid.layers.concat([prev_crop, up_sample], axis=1)
            x = seq_layer(out)

        return x


class UNet(dygraph.Layer):
    """
        4 downsample.
        4 upsample.
    """

    def __init__(self, num_inp=3, num_out=59, act="relu"):
        super(UNet, self).__init__()
        self.encoder = Encoder(num_inp)
        self.middle_layer = Sequential(
            Conv2D(512, 1024, filter_size=3, stride=1, padding=0, act=None),
            BatchNorm(1024, act=act),
            Conv2D(1024, 1024, filter_size=3, stride=1, padding=0, act=None),
            BatchNorm(1024, act=act),
        )
        self.decoder = Decoder()

        self.header = Sequential(
            Conv2D(64, num_out, filter_size=1, stride=1, padding=0, act=None),
            BatchNorm(num_out, act=None)
        )

    def forward(self, inp):
        (x, pool_layers) = self.encoder(inp)
        x = self.middle_layer(x)
        x = self.decoder(x, pool_layers)
        return self.header(x)


if __name__ == "__main__":
    from paddle.fluid.dygraph import to_variable
    import numpy as np
    inp = np.random.normal(size=(1, 3, 572, 572)).astype("float32")

    with dygraph.guard():
        model = UNet()
        model.eval()

        inp = to_variable(inp)
        out = model(inp)
        print(f"inp shape {inp.shape}")
        print(f"out shape {out.shape}")
