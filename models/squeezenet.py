from torch import nn

import ai8x
import ai8x_blocks


class SqueezeNet(nn.Module):
    """
    SqueezeNet for AI85.
    The last layer is implemented as a linear layer rather than a convolution
    layer as defined in th eoriginal paper.
    """

    def __init__(
        self,
        num_channels=3,
        num_classes=10,
        dimensions=(224, 224),
        bias=False,
        **kwargs,
    ):
        super().__init__()
        dim1 = dimensions[0]
        dim2 = dimensions[1]
        # 3x224x224
        self.conv1 = ai8x.FusedConv2dReLU(
            in_channels=num_channels,
            out_channels=96,
            kernel_size=3,
            padding=1,
            bias=bias,
            **kwargs,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        dim1 //= 2
        dim2 //= 2
        # 96x112x112
        self.fire2 = ai8x_blocks.Fire(
            in_planes=96,
            squeeze_planes=16,
            expand1x1_planes=64,
            expand3x3_planes=64,
            bias=bias,
            **kwargs,
        )
        # 128x112x112
        self.fire3 = ai8x_blocks.Fire(
            in_planes=128,
            squeeze_planes=16,
            expand1x1_planes=64,
            expand3x3_planes=64,
            bias=bias,
            **kwargs,
        )

        self.add1 = ai8x.Add()

        self.fire4 = ai8x_blocks.Fire(
            in_planes=128,
            squeeze_planes=32,
            expand1x1_planes=128,
            expand3x3_planes=128,
            bias=bias,
            **kwargs,
        )
        # 256x112x112
        self.pool2 = nn.MaxPool2d(
            kernel_size=4, stride=4, padding=0
        )  # check if kernel size=3
        dim1 //= 4
        dim2 //= 4
        # 256x28x28
        self.fire5 = ai8x_blocks.Fire(
            in_planes=256,
            squeeze_planes=32,
            expand1x1_planes=128,
            expand3x3_planes=128,
            bias=bias,
            **kwargs,
        )

        self.add2 = ai8x.Add()

        # 256x28x28
        self.fire6 = ai8x_blocks.Fire(
            in_planes=256,
            squeeze_planes=48,
            expand1x1_planes=192,
            expand3x3_planes=192,
            bias=bias,
            **kwargs,
        )
        # 384x28x28
        self.fire7 = ai8x_blocks.Fire(
            in_planes=384,
            squeeze_planes=64,
            expand1x1_planes=192,
            expand3x3_planes=192,
            bias=bias,
            **kwargs,
        )

        self.add3 = ai8x.Add()

        self.fire8 = ai8x_blocks.Fire(
            in_planes=384,
            squeeze_planes=64,
            expand1x1_planes=256,
            expand3x3_planes=256,
            bias=bias,
            **kwargs,
        )
        # 512x28x28
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0
        )  # check if kernel size=3
        dim1 //= 2
        dim2 //= 2
        # 512x14x14
        self.fire9 = ai8x_blocks.Fire(
            in_planes=512,
            squeeze_planes=64,
            expand1x1_planes=256,
            expand3x3_planes=256,
            bias=bias,
            **kwargs,
        )

        self.add4 = ai8x.Add()

        # 512x14x14
        self.conv10 = ai8x.FusedAvgPoolConv2dAbs(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=1,
            pool_size=14,
            pool_stride=14,
        )
        # self.fc = ai8x.SoftwareLinear(512*dim1*dim2, num_classes, bias=bias)
        # num_classesx1x1

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.pool1(x)
        resid = self.fire2(x)
        x = self.fire3(resid)
        x = self.add1(x, resid)
        x = self.fire4(x)

        resid = self.pool2(x)
        x = self.fire5(resid)
        x = self.add2(x, resid)
        resid = self.fire6(x)
        x = self.fire7(resid)
        x = self.add3(x, resid)
        x = self.fire8(x)

        resid = self.pool3(x)
        x = self.fire9(resid)
        x = self.add4(x, resid)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


def squeezenet(pretrained=False, **kwargs):
    """
    Constructs a AI85SqueezeNet model.
    """
    assert not pretrained
    return SqueezeNet(**kwargs)


models = [
    {
        "name": "squeezenet",
        "min_input": 1,
        "dim": 2,
    },
]
