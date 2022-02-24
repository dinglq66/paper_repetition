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
