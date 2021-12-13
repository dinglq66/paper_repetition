from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


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
    def __init__(self, drop_prob=0., training=False):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    将二维图像块编码为张量格式，初始化方法中的参数与原论文保持一致
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size  # img_size表示输入图像大小
        self.patch_size = patch_size  # patch_size表示将图像划分为子块的大小
        # 根据img_size和patch_size可以计算出切分出的网格形状以及网格总数
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 定义一个卷积层，输入维度是3（彩色图像），输出维度是embed_dim（默认值为768）
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 定义正则化层，如果未指定正则化方式则采用恒等映射
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] --> [B, C, HW]
        # transpose: [B, C, HW] --> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """
    多头注意力机制，与原论文中保持一致，这里采用8头
    """
    def __init__(self,
                 dim,  # 输入token的维度
                 num_heads=8,  # 多头注意力的个数
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 用总维数除以注意力头的个数即可得到每个注意力头的维度(地板除)
        head_dim = dim // num_heads
        # 若未指定qk_scale则除以根号head_dim，本质上是为了使随机变量的方差变为1，推导过程根据qxk随机变量的均值方差得到
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim], 第二个维度加一是因为添加了class的编码
        B, N, C = x.shape

        # qkv(): --> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: --> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: --> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 将q、k、v分离开，各自的维度都是[batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: --> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # @: multiply --> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply --> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: --> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: --> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    全连接层，和多头注意力模块共同组成基本Block
    """
    def __init__(self):
        super().__init__()
        
