import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np
from typing import Optional


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    该函数的作用在于，在残差结构中使用Stochastic Depth，类似于Dropout，即对残差部分（区别于恒等映射部分）进行随机丢弃，
    同时要保持结果均值为1
    :param x:残差部分输出的张量
    :param drop_prob:丢弃概率
    :param training:是否为训练阶段（随机深度网络只在训练阶段使用）
    :return:
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, ..., 1)，共有x.ndim-1个1
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 取小于等于每个值的最大整数，将张量的值控制在0~1范围内
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    随机深度网络对应的类，实例属性包括drop_prob和training，前者用于表示随机丢弃的概率，后者用于表示是否为训练阶段
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将特征图按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口尺寸(M)

    Returns:
         windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, C] --> [B, H//Mh, W/Mw, Mh, Mw, C]
    # view: [B, H//Mh, W/Mw, Mh, Mw, C] --> [B * num_windows, Mh, Mw, C]
    # contiguous()函数用于断开windows变量和x变量之间的关系，和深拷贝相似
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将windows还原成一个特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 图像的高度
        W (int): 图像的宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] --> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] --> [B, H//Mh, Mh, W//Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, C] --> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    将二维图像块编码为张量格式，实现方法与ViT类似，在将图像块到向量的映射过程中通过一个卷积层实现
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图像的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # 填充最后三个维度，填充时是按照x张量形状最后三个维度的顺序进行的，且从后向前，因此在使用pad函数时要注意x.shape
            # (W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] --> [B, C, HW]
        # transpose: [B, C, HW] --> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """
    Patch Merging层实现，实现特征图维度的减半，从而达到多尺度的目的
    Args:
        dim (int): 输入部分的通道数
        norm_layer (nn.Module, optional): 正则化层，默认为nn.LayerNorm()
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 != 0) or (W % 2 != 0)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        return x


class MLP(nn.Module):
    """
    全连接层，和ViT中的MLP一致
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    """

    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义一个变量用于存储相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)  # [2*Mh-1 * 2*Mw-1, num_heads]
        )

        # 对窗口中的每一个token得到pair-wise的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # indexing="ij": 按照ij坐标系建立网格，其中纵向表示i轴，横向表示j轴
        # 若不指定indexing参数，建立的网格坐标系为笛卡尔坐标系，纵向表示y轴，横向表示x轴
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]

        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_coords_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        # register_buffer类型的参数不会在optimizer.step()中更新，只会在forward过程中更新；网络存储时会将其中的参数也保存下来
        self.register_buffer("relative_position_index", relative_coords_index)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: --> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: --> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q/k/v: [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]


