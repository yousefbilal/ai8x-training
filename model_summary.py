from torch import nn
import ai8x
import ai8x_blocks
# import torchsummary
import torchinfo

class AI87MobileNetV2(nn.Module):
    """
    MobileNet v2 for MAX78002
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pre_layer_stride,
        bottleneck_settings,
        last_layer_width,
        avg_pool_size=4,
        num_classes=100,
        num_channels=3,
        dimensions=(32, 32),  # pylint: disable=unused-argument
        bias=False,
        depthwise_bias=False,
        **kwargs,
    ):
        super().__init__()

        self.pre_stage = ai8x.FusedConv2dBNReLU(
            num_channels,
            bottleneck_settings[0][1],
            3,
            padding=1,
            stride=pre_layer_stride,
            bias=bias,
            **kwargs,
        )

        self.feature_stage = nn.ModuleList([])
        for setting in bottleneck_settings:
            self._create_bottleneck_stage(setting, bias, depthwise_bias, **kwargs)

        self.post_stage = ai8x.FusedConv2dReLU(
            bottleneck_settings[-1][2],
            last_layer_width,
            1,
            padding=0,
            stride=1,
            bias=False,
            **kwargs,
        )

        self.classifier = ai8x.FusedAvgPoolConv2d(
            last_layer_width,
            num_classes,
            1,
            padding=0,
            stride=1,
            pool_size=avg_pool_size,
            pool_stride=avg_pool_size,
            bias=False,
            wide=True,
            **kwargs,
        )

    def _create_bottleneck_stage(self, setting, bias, depthwise_bias, **kwargs):
        """Function to create bottleneck stage. Setting format is:
        [num_repeat, in_channels, out_channels, stride, expansion_factor]
        """
        stage = []

        if setting[0] > 0:
            stage.append(
                ai8x_blocks.ResidualBottleneck(
                    in_channels=setting[1],
                    out_channels=setting[2],
                    stride=setting[3],
                    expansion_factor=setting[4],
                    bias=bias,
                    depthwise_bias=depthwise_bias,
                    **kwargs,
                )
            )

            for _ in range(1, setting[0]):
                stage.append(
                    ai8x_blocks.ResidualBottleneck(
                        in_channels=setting[2],
                        out_channels=setting[2],
                        stride=1,
                        expansion_factor=setting[4],
                        bias=bias,
                        depthwise_bias=depthwise_bias,
                        **kwargs,
                    )
                )

        self.feature_stage.append(nn.Sequential(*stage))

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.pre_stage(x)
        for stage in self.feature_stage:
            x = stage(x)
        x = self.post_stage(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        
def mymobilenet(pretrained=False):
    """
    Constructs a MobileNet v2 model for Cifar-100 dataset optimized for AI87.
    """
    assert not pretrained
    # settings for bottleneck stages in format
    # [num_repeat, in_channels, out_channels, stride, expansion_factor]
    # Updated bottleneck settings
    # bottleneck_settings = [
    #     [1, 16, 4, 1, 1],  # Stage 1: No reduction
    #     [2, 4, 24, 2, 6],  # Stage 2: Reduction 112 -> 56
    #     [3, 24, 32, 2, 6],  # Stage 3: Reduction 56 -> 28
    #     [4, 32, 56, 2, 6],  # Stage 4: Reduction 28 -> 14
    #     [3, 56, 56, 1, 6],  # Stage 5: No reduction
    #     [3, 56, 64, 2, 6], # Stage 6: Reduction 14 -> 7
    #     [1, 64, 128, 1, 6],# Final stage: No reduction
    # ]

    bottleneck_settings = [
        [1, 16, 8, 1, 1],
        [2, 8, 12, 2, 6],
        [3, 12, 16, 2, 6],
        [4, 16, 32, 2, 6],
        [3, 32, 48, 2, 6],
        [3, 48, 80, 2, 6],
        [1, 80, 160, 1, 6],
    ]
    
    return AI87MobileNetV2(
        pre_layer_stride=1,
        bottleneck_settings=bottleneck_settings,
        last_layer_width=256,
        avg_pool_size=7,
        depthwise_bias=False,
        num_classes=2,
        num_channels=3,
        dimensions=(224, 224),
        bias=True,
    )

def ai87netmobilenetv2cifar100_m0_5(pretrained=False):
    """
    Constructs a MobileNet v2 model for Cifar-100 dataset optimized for AI87.
    """
    assert not pretrained
    # settings for bottleneck stages in format
    # [num_repeat, in_channels, out_channels, stride, expansion_factor]
    bottleneck_settings = [
        [1, 16, 8, 1, 1],
        [2, 8, 12, 2, 6],
        [3, 12, 16, 2, 6],
        [4, 16, 32, 2, 6],
        [3, 32, 48, 1, 6],
        [3, 48, 80, 1, 6],
        [1, 80, 160, 1, 6],
    ]

    return AI87MobileNetV2(
        pre_layer_stride=1,
        bottleneck_settings=bottleneck_settings,
        last_layer_width=640,
        avg_pool_size=4,
        depthwise_bias=True,
        num_channels=3,
        dimensions=(224, 224),
        bias=True,
        num_classes=100,
    )

if __name__ == '__main__':
    ai8x.set_device(
        87,
        False,
        False,
        verbose=True,
    )
    # mymodel = ai87netmobilenetv2cifar100_m0_5()
    # torchinfo.summary(mymodel, (1, 3, 32, 32), device='cpu', depth=4, col_names=["input_size", "output_size", "num_params", "params_percent"], mode="eval")
    mymodel = mymobilenet()
    torchinfo.summary(mymodel, (1, 3, 224, 224), device='cpu', depth=4, col_names=["input_size", "output_size", "num_params", "params_percent"], mode="eval")

