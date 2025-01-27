from torch import nn

import ai8x
import ai8x_blocks


class SqueezeNet(nn.Module):

    def __init__(
        self,
        num_channels=3,
        num_classes=2,
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
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=bias,
            **kwargs,
        )
        # 64x224x224
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        dim1 //= 2
        dim2 //= 2
        # 64x112x112
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        dim1 //= 2 - 1
        dim2 //= 2 - 1
        # 64x55x55
        self.fire2 = ai8x_blocks.Fire(
            in_planes=64,
            squeeze_planes=16,
            expand1x1_planes=64,
            expand3x3_planes=64,
            bias=bias,
            **kwargs,
        )
        # 128x55x55
        self.fire3 = ai8x_blocks.Fire(
            in_planes=128,
            squeeze_planes=16,
            expand1x1_planes=64,
            expand3x3_planes=64,
            bias=bias,
            **kwargs,
        )
        # 128x55x55
        self.pool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0
        )
        dim1 //= 2
        dim2 //= 2
        # 128x27x27
        self.fire4 = ai8x_blocks.Fire(
            in_planes=128,
            squeeze_planes=32,
            expand1x1_planes=128,
            expand3x3_planes=128,
            bias=bias,
            **kwargs,
        )
        # 256x27x27
        self.fire5 = ai8x_blocks.Fire(
            in_planes=256,
            squeeze_planes=32,
            expand1x1_planes=128,
            expand3x3_planes=128,
            bias=bias,
            **kwargs,
        )
        # 256x27x27
        self.pool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0
        )
        dim1 //= 2
        dim2 //= 2
        # 256x13x13
        self.fire6 = ai8x_blocks.Fire(
            in_planes=256,
            squeeze_planes=48,
            expand1x1_planes=192,
            expand3x3_planes=192,
            bias=bias,
            **kwargs,
        )
        # 384x13x13
        self.fire7 = ai8x_blocks.Fire(
            in_planes=384,
            squeeze_planes=48,
            expand1x1_planes=192,
            expand3x3_planes=192,
            bias=bias,
            **kwargs,
        )
        # 384x13x13
        self.fire8 = ai8x_blocks.Fire(
            in_planes=384,
            squeeze_planes=64,
            expand1x1_planes=256,
            expand3x3_planes=256,
            bias=bias,
            **kwargs,
        )
        # 512x13x13
        self.fire9 = ai8x_blocks.Fire(
            in_planes=512,
            squeeze_planes=64,
            expand1x1_planes=256,
            expand3x3_planes=256,
            bias=bias,
            **kwargs,
        )
        self.drop = nn.Dropout(0.5)
        # 512x13x13
        self.conv10 = ai8x.FusedAvgPoolConv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=1,
            pool_size=13,
            pool_stride=13,
        )
        # self.fc = ai8x.SoftwareLinear(512*dim1*dim2, num_classes, bias=bias)
        # num_classesx1x1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.pool0(x)
        x = self.pool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool2(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.pool3(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)
        x = self.drop(x)
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
