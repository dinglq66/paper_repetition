import torch
import torch.nn as nn


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    该函数的目的在于确保网络中所有层的通道数都能被8整除
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 确保减少的通道数不超过原始通道数的10%
    if new_ch < 0.9*ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channels * expand_ratio
        self.use_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1x1 PW（点卷积）
            layers.append(ConvBNReLU(in_channels=in_channels, out_channels=hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 DW（深度卷积）
            ConvBNReLU(in_channels=hidden_channel, out_channels=hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 线性PW
            nn.Conv2d(in_channels=hidden_channel, out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(ch=32 * alpha, divisor=round_nearest)
        last_channel = _make_divisible(ch=1280 * alpha, divisor=round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        # first conv1 layer
        features = [ConvBNReLU(in_channels=3, out_channels=input_channel, stride=2)]

        # 添加BottleNeck部分的网络
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channels=input_channel, out_channels=output_channel,
                                      stride=stride, expand_ratio=t))
                input_channel = output_channel
        # 搭建最后的1x1卷积层
        features.append(ConvBNReLU(in_channels=input_channel, out_channels=last_channel, kernel_size=1))
        # 整合features层
        self.features = nn.Sequential(*features)

        # 搭建分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MobileNetV2()
    test_tensor = torch.ones((1, 3, 224, 224))
    print(model(test_tensor).shape)
