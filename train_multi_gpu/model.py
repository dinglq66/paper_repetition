import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    用于ResNet18/34结构的基本模块
    """
    expansion = 1  # 模块中最后一层卷积层通道数相比于第一层卷积层通道数的比例

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    用于ResNet50/101结构的基本模块
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channels * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width,
                               kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # ---------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=(3, 3), stride=(stride, stride), bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # ---------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channels*self.expansion,
                               kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # 彩色对象来说，conv1的in_channels为3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels,
                               kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels=channel*block.expansion,
                          kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )

        layers = [block(self.in_channels, channel, downsample=downsample,
                        stride=stride, groups=self.groups, width_per_group=self.width_per_group)]
        self.in_channels = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channel, groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    return ResNet(block=BasicBlock,
                  blocks_num=[2, 2, 2, 2],
                  num_classes=num_classes,
                  include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(block=BasicBlock,
                  blocks_num=[3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck,
                  blocks_num=[3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck,
                  blocks_num=[3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNet(block=Bottleneck,
                  blocks_num=[3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 8
    return ResNet(block=Bottleneck,
                  blocks_num=[3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


if __name__ == '__main__':
    # model = resnet18()
    model = resnext101_32x8d()
    test_tensor = torch.ones((1, 3, 224, 224))
    print(model(test_tensor).shape)
