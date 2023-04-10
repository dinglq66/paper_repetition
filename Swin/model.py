import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import torch.utils.checkpoint as checkpoint

import math
import einops
import numpy as np
from thop import profile


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


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


class ImageToTokens(nn.Module):
    def __init__(self, patch_size=4, in_c=1, embed_dim=48, norm_layer=None):
        super(ImageToTokens, self).__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.compress = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=self.patch_size,
                                  stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # [B, 1, H, W]
        _, _, H, W = x.shape
        nH, nW = H // self.patch_size[0], W // self.patch_size[1]

        # padding
        # 如果输入图像的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # 填充最后三个维度，填充时是按照x张量形状最后三个维度的顺序进行的，且从后向前，因此在使用pad函数时要注意x.shape
            # (W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))
        # [B, 1, H, W] -> [B, C, H, W]
        x = self.proj(x)
        # [B, C, H, W] -> [B, C, H // patch, W // patch] = [B, C, nH, nW]
        x = self.compress(x)
        # [B, C, nH, nW] -> [B, nH * nW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, nH, nW


class DeformableAttention(nn.Module):
    def __init__(self, fmap_size, embed_dim, n_heads, n_head_channels, stride, offset_range_factor=-1, attn_drop=0.,
                 proj_drop=0., norm_layer=None, use_pe=False, dwc_pe=False, fixed_pe=False):
        super(DeformableAttention, self).__init__()
        assert embed_dim == n_heads * n_head_channels, \
            'Error! Notice the equation of embed_dim and n_heads * n_head_channels'
        self.fmap_size = fmap_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.stride = stride
        self.offset_range_factor = offset_range_factor
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.fixed_pe = fixed_pe

        self.proj_v = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1), groups=embed_dim)
        self.proj_q = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=embed_dim)
        self.proj_k = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                                kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), groups=embed_dim)
        self.proj_off = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(stride, stride),
                      padding=(1, 1), groups=embed_dim),
            LayerNormProxy(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=2, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False)
        )
        self.proj_out = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1), stride=(1, 1),
                                  padding=(0, 0), bias=False)
        self.attn_drop = DropPath(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj_drop = DropPath(proj_drop) if proj_drop > 0. else nn.Identity()

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                                           kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=embed_dim)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.fmap_size * self.fmap_size, (self.fmap_size // self.stride)**2)
                )
                nn.init.trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, 2 * (self.fmap_size // self.stride) - 1,
                                2 * (self.fmap_size // self.stride) - 1)
                )
                nn.init.trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H, W, B, data_type, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=data_type, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=data_type, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W).mul_(2).sub_(1)
        ref[..., 0].div_(H).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B, -1, -1, -1)
        return ref

    def forward(self, x):
        B, C, H, W = x.shape
        data_type, device = x.dtype, x.device

        # value: [B, C, H, W]
        value = self.proj_v(x)
        # query: [B, C, H, W] -> [B * num_heads, embed_dim_per_head, H*W]
        query = self.proj_q(x).reshape(B*self.n_heads, self.n_head_channels, -1)
        # key: [B, C, H, W] -> [B * num_heads, embed_dim_per_head, Hk*Wk]
        key = self.proj_k(x).reshape(B*self.n_heads, self.n_head_channels, -1)
        # offset: [B, 2, Hk, Wk]
        offset = self.proj_off(x)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # offset: [B, 2, Hk, Wk] -> [B, Hk, Wk, 2]
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # reference: [B, Hk, Wk, 2]
        reference = self._get_ref_points(Hk, Wk, B, data_type, device)

        # pos: [B, Hk, Wk, 2]
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        # v_sampled: [B, C, Hk, Wk]
        v_sampled = F.grid_sample(
            input=value,
            grid=pos[..., (1, 0)],
            # grid=pos,
            mode='bilinear', align_corners=True
        )
        # v_sampled: [B, C, Hk, Wk] -> [B * num_heads, embed_dim_per_head, Hk * Wk]
        v_sampled = v_sampled.reshape(B*self.n_heads, self.n_head_channels, -1)
        # attn: [B * num_heads, H * W, Hk * Wk]
        attn = torch.einsum('b c m, b c n -> b m n', query, key)
        attn = attn.mul(self.scale)

        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(query.reshape(B, C, H, W)).reshape(
                    B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn += attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_ref_points(H, W, B, data_type, device)

                displacement = (q_grid.reshape(B, H * W, 2).unsqueeze(2) -
                                pos.reshape(B, n_sample, 2).unsqueeze(1)).mul(0.5)
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B, self.n_heads, 2 * Hk - 1, 2 * Wk - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v_sampled)
        if self.use_pe and self.dwc_pe:
            out += residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y


class MixFeedForward(nn.Module):
    def __init__(self, embed_dim, ratio=2, act_layer=nn.GELU, dropout=0.):
        super(MixFeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * ratio
        self.ratio = ratio
        self.act_layer = act_layer
        self.drop_1 = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.drop_2 = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.fc = nn.Linear(in_features=embed_dim, out_features=self.hidden_dim)
        self.LocalEnhance = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), groups=self.hidden_dim),
            LayerNormProxy(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.embed_dim, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0))
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        x = self.fc(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.drop_1(x)
        # [B, C*ratio, H, W] -> [B, C, H, W]
        x = self.LocalEnhance(x)
        x = self.drop_2(x)
        return x


class TransformBlock(nn.Module):
    def __init__(self, fmap_size, embed_dim, n_heads, n_head_channels, stride, offset_range_factor=-1, ratio=2,
                 attn_drop=0., proj_drop=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=LayerNormProxy,
                 use_pe=False, dwc_pe=False):
        super(TransformBlock, self).__init__()
        self.norm1 = norm_layer(embed_dim)
        self.drop = DropPath(drop_prob=drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        self.attn = DeformableAttention(fmap_size=fmap_size, embed_dim=embed_dim, n_heads=n_heads,
                                        n_head_channels=n_head_channels, stride=stride,
                                        offset_range_factor=offset_range_factor,
                                        attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
                                        use_pe=use_pe, dwc_pe=dwc_pe)
        self.ffn = MixFeedForward(embed_dim=embed_dim, ratio=ratio, act_layer=act_layer, dropout=drop_path_ratio)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class Downsampling(nn.Module):
    def __init__(self, dim, norm_layer=LayerNormProxy):
        super(Downsampling, self).__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=(1, 1), stride=(1, 1),
                                   padding=(0, 0), bias=False)
        self.norm = LayerNormProxy(2 * dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
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
        x = self.norm(x.permute(0, 3, 1, 2))
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, fmap_size, embed_dim, depth, num_heads, stride, offset_range_factor=-1, ratio=2,
                 attn_drop=0., proj_drop=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=LayerNormProxy,
                 downsample=False, use_pe=False, dwc_pe=False):
        super(BasicLayer, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.stride = stride

        self.blocks = nn.ModuleList([
            TransformBlock(fmap_size=fmap_size, embed_dim=embed_dim, n_heads=num_heads,
                           n_head_channels=embed_dim // num_heads, stride=stride,
                           offset_range_factor=offset_range_factor,
                           ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop, drop_path_ratio=drop_path_ratio,
                           act_layer=act_layer, norm_layer=norm_layer, use_pe=use_pe, dwc_pe=dwc_pe)
            for i in range(depth)
        ])
        if downsample:
            self.downsample = Downsampling(dim=embed_dim, norm_layer=LayerNormProxy)
        else:
            self.downsample = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.downsample:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class Conv_Former(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_c=1, embed_dim=48, depths=(3, 3, 4), num_heads=(3, 6, 12),
                 stride=2, offset_range_factor=[-1, 2, 2], ratio=2, attn_drop=0., proj_drop=0., drop_path_rate=0.1,
                 norm_layer=LayerNormProxy, use_pe=[False, True, True], dwc_pe=[False, False, False]):
        super(Conv_Former, self).__init__()
        self.fmap_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.use_pe = use_pe
        self.dwc_pe =dwc_pe
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.patch_embed = ImageToTokens(patch_size=patch_size, in_c=in_c, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        self.fmap_size = self.fmap_size // patch_size
        self.stage1 = BasicLayer(fmap_size=self.fmap_size, embed_dim=embed_dim, depth=depths[0], num_heads=num_heads[0],
                                 stride=stride, offset_range_factor=offset_range_factor[0],
                                 ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                                 drop_path_ratio=dpr[0], downsample=False, use_pe=use_pe[0], dwc_pe=dwc_pe[0])

        self.layers = nn.ModuleList([
            BasicLayer(fmap_size=self.fmap_size // (2**(i+1)), embed_dim=int(embed_dim * 2 ** (i+1)), depth=depths[i+1],
                       num_heads=num_heads[i+1], stride=stride, offset_range_factor=offset_range_factor[i+1],
                       ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                       drop_path_ratio=dpr[i+1], downsample=True, use_pe=use_pe[i+1], dwc_pe=dwc_pe[i+1])
            for i in range(self.num_layers-1)
        ])
        self.norm = norm_layer(self.num_features)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        x, nH, nW = self.patch_embed(x)
        x = x.reshape(B, nH, nW, -1).permute(0, 3, 1, 2)
        feature_map = []

        x = self.stage1(x)
        feature_map.append(x)

        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        x = self.norm(x)
        return x, feature_map


class ISODformer(nn.Module):
    def __init__(self, mode='non-local', img_size=256, patch_size=4, in_c=1, embed_dim=96, depths=(3, 3, 4),
                 num_heads=(3, 6, 12), stride=2, offset_range_factor=[-1, 2, 2],
                 ratio=2, attn_drop=0., proj_drop=0., drop_path_rate=0.1, norm_layer=LayerNormProxy):
        super(ISODformer, self).__init__()
        self.mode = mode
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.backbone = Conv_Former(img_size=img_size,
                                    patch_size=patch_size,
                                    in_c=in_c,
                                    embed_dim=embed_dim,
                                    depths=depths,
                                    num_heads=num_heads,
                                    stride=stride,
                                    offset_range_factor=offset_range_factor,
                                    ratio=ratio,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    drop_path_rate=drop_path_rate,
                                    norm_layer=norm_layer
                                    )
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features // 4,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(self.num_features // 4),
            LayerNormProxy(self.num_features // 4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features // 2, out_channels=self.num_features // 4,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(self.num_features // 4),
            LayerNormProxy(self.num_features // 4),
            nn.ReLU(inplace=True)
        )
        if self.mode == 'non-local':
            self.fuse1 = FuseBlock(in_channels=self.num_features // 4)
            self.fuse2 = FuseBlock(in_channels=self.num_features // 4)
        else:
            raise NotImplementedError

        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channels=self.num_features // 4, out_channels=self.num_features // 4,
        #               kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
        #     # nn.BatchNorm2d(self.num_features // 4),
        #     LayerNormProxy(self.num_features // 4),
        #     # 采用ALCNet中的FCNHead
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Conv2d(in_channels=self.num_features // 4, out_channels=1,
        #               kernel_size=(1, 1), stride=(1, 1), bias=False)
        # )

        self.norm = nn.BatchNorm2d(self.num_features // 4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features // 4, 5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        map3, feature_maps = self.backbone(x)
        map1, map2 = feature_maps[0], feature_maps[1]
        map3_dec = self.dec3(map3)
        map2_dec = self.dec2(map2)
        if self.mode == 'non-local':
            up_map3 = F.interpolate(map3_dec, scale_factor=2, mode='bilinear', align_corners=True)
            map23 = self.fuse1(map2_dec, up_map3)  # non-local
            up_map23 = F.interpolate(map23, scale_factor=2, mode='bilinear', align_corners=True)
            map_123 = self.fuse2(map1, up_map23)  # non-local
        else:
            raise NotImplementedError

        # segmentation
        # out = self.head(map_123)
        # out = F.interpolate(out, scale_factor=self.patch_size, mode='bilinear', align_corners=True)

        # classification
        out = self.norm(map_123)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.head(out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, (1, 1))
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, (1, 1))
        self.conv_d = nn.Conv2d(in_channels, in_channels, (1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))  # 对有batch的3维度张量进行相乘运算
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class FuseBlock(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(FuseBlock, self).__init__()
        self.sam = SpatialAttention(in_channels)
        self.cam = ChannelAttention()
        self.end_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_low, x_high):
        spatial_attn = self.sam(x_low)
        channel_attn = self.cam(x_high)

        feature = x_low + x_high
        feature = self.end_conv(feature)
        feature = feature * spatial_attn
        feature = feature * channel_attn
        return feature


# DAT model
class LocalAttention(nn.Module):
    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super(LocalAttention, self).__init__()
        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0, 'dim can not % number of heads'
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c',
                                   h1=self.window_size[0], w1=self.window_size[1])  # B, Nr, Ws, C
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total)  # B', N, 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn: (b * nW) h w w
            # mask: nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x))  # B', N, C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2,
                             h1=self.window_size[0], w1=self.window_size[1])  # B, C, H, W
        return x, None, None


class ShiftWindowAttention(LocalAttention):
    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        super(ShiftWindowAttention, self).__init__(dim, heads, window_size, attn_drop, proj_drop)
        self.fmap_size = to_2tuple(window_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)  # H, W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)',
                                        h1=self.window_size[0], w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x, None, None


class DAttentionBaseline(nn.Module):
    def __init__(self, q_size, kv_size, n_heads, n_head_channels, n_groups,
                 attn_drop, proj_drop, stride,
                 offset_range_factor, use_pe, dwc_pe,
                 no_off, fixed_pe, stage_idx):
        super(DAttentionBaseline, self).__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, (kk, kk), stride, (kk // 2, kk // 2),
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, (1, 1), (1, 1), (0, 0), bias=False)
        )
        self.proj_q = nn.Conv2d(
            self.nc, self.nc, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.proj_k = nn.Conv2d(
            self.nc, self.nc, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.proj_v = nn.Conv2d(
            self.nc, self.nc, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.proj_out = nn.Conv2d(
            self.nc, self.nc, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc,
                                           kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=self.nc)
            elif self.fixed_pe:
                self.rep_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B*g, H, W, 2

        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # B*g, 2, Hg, Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],
            mode='bilinear', align_corners=True
        )  # B*g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B*h, hW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(
                    B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_ref_points(H, W, B, dtype, device)

                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) -
                                pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B*g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

            attn = F.softmax(attn, dim=2)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b c n -> b c m', attn, v)

            if self.use_pe and self.dwc_pe:
                out = out + residual_lepe
            out = out.reshape(B, C, H, W)

            y = self.proj_drop(self.proj_out(out))

            return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class MLP(nn.Module):
    def __init__(self, channels, expansion, drop):
        super(MLP, self).__init__()
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):
        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, (1, 1), (1, 1), 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, (1, 1), (1, 1), 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, (3, 3), (1, 1), 1, groups=self.dim2)

    def forward(self, x):
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))

        return x


class TransformerStage(nn.Module):
    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):
        super(TransformerStage, self).__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, (1, 1), (1, 1), (0, 0)) if dim_in != dim_embed else nn.Identity()
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop)
                if use_dwc_mlp else MLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )

        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.proj(x)
        positions = []
        references = []
        for d in range(self.depths):
            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references


class DAT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 strides=[-1, -1, -1, -1], offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 **kwargs
                 ):
        super(DAT, self).__init__()
        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, (7, 7), (patch_size, patch_size), 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, (patch_size, patch_size), (patch_size, patch_size), 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                                 dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i],
                                 sr_ratios[i], heads[i], strides[i],
                                 offset_range_factor[i], i,
                                 dwc_pes[i], no_offs[i], fixed_pes[i],
                                 attn_drop_rate, drop_rate, expansion, drop_rate,
                                 dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                 use_dwc_mlps[i])
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], (3, 3), (2, 2), (1, 1), bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], (2, 2), (2, 2), (0, 0), bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.cls_norm = LayerNormProxy(dims[-1])
        self.cls_head = nn.Linear(dims[-1], num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pertrained(self, state_dict):

        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_value] = state_value
            else:
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)

        self.load_state_dict(new_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        x = self.patch_proj(x)
        positions = []
        references = []
        feature_map = []
        for i in range(4):
            x, pos, ref = self.stages[i](x)
            feature_map.append(x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.cls_head(x)

        return x, positions, references, feature_map


class DATISOD(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, depths=(2, 2, 6, 2)):
        super(DATISOD, self).__init__()
        self.backbone = DAT(img_size=256, patch_size=4, num_classes=1000, expansion=4,
                            dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                            heads=[3, 6, 12, 24],
                            window_sizes=[8, 8, 8, 8],
                            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2,
                            strides=[-1, -1, 1, 1], offset_range_factor=[-1, -1, 2, 2],
                            stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                            groups=[-1, -1, 3, 6],
                            use_pes=[False, False, True, True],
                            dwc_pes=[False, False, False, False],
                            sr_ratios=[-1, -1, -1, -1],
                            fixed_pes=[False, False, False, False],
                            no_offs=[False, False, False, False],
                            ns_per_pts=[4, 4, 4, 4],
                            use_dwc_mlps=[False, False, False, False],
                            use_conv_patches=False)
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features // 2, out_channels=self.num_features // 8,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(self.num_features // 8),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features // 4, out_channels=self.num_features // 8,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(self.num_features // 8),
            nn.ReLU(inplace=True)
        )

        # self.attn_1 = BottomUpLocal_FPNFuse(channels=self.num_features // 4)  # BottomToUp
        # self.attn_2 = BottomUpLocal_FPNFuse(channels=self.num_features // 4)  # BottomToUp

        self.attn1 = FuseBlock(in_channels=self.num_features // 8)  # non-local
        self.attn2 = FuseBlock(in_channels=self.num_features // 8)  # non-local

        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channels=self.num_features // 8, out_channels=self.num_features // 8,
        #               kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
        #     nn.BatchNorm2d(self.num_features // 8),
        #     # 采用ALCNet中的FCNHead
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Conv2d(in_channels=self.num_features // 8, out_channels=1,
        #               kernel_size=(1, 1), stride=(1, 1), bias=False)
        # )

        self.norm = nn.BatchNorm2d(self.num_features // 8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features // 8, 5)

    def forward(self, x):
        B, C, H, W = x.shape
        _, _, _, feature_map = self.backbone(x)
        map1, map2, map3 = feature_map[0], feature_map[1], feature_map[2]
        # map1 = map1.reshape(B, H // 4, W // 4, -1).permute(0, 3, 1, 2)
        # map2 = map2.reshape(B, H // 8, W // 8, -1).permute(0, 3, 1, 2)
        # map3 = map3.reshape(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
        # print(map1.shape, map2.shape, map3.shape)
        map3_dec = self.dec3(map3)
        map2_dec = self.dec2(map2)
        up_map3 = F.interpolate(map3_dec, scale_factor=2, mode='bilinear', align_corners=True)
        # map23 = self.attn_1(up_map3, map2_dec)  # BottomToUp
        map23 = self.attn1(map2_dec, up_map3)  # non-local
        up_map23 = F.interpolate(map23, scale_factor=2, mode='bilinear', align_corners=True)
        # map_123 = self.attn_2(up_map23, map1)  # BottomToUp
        map_123 = self.attn2(map1, up_map23)  # non-local
        # out = self.head(map_123)
        # out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.norm(map_123)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.head(out)
        return out


if __name__ == '__main__':
    # input_tensor = torch.ones(size=(2, 48, 64, 64))
    # Deformable_attn = DAttentionBase(q_size=64, kv_size=64, n_heads=4, n_head_channels=12, n_groups=8,
    #                                  attn_drop=0.2, proj_drop=0.2, stride=1, stage_idx=3)
    # result, pos, ref = Deformable_attn(input_tensor)
    # print('shape of result:{}, shape of pos:{}, shape of ref:{}'.format(result.shape, pos.shape, ref.shape))
    input_tensor = torch.ones(size=(2, 1, 256, 256))
    patch_embedding = ImageToTokens(patch_size=4, in_c=1, embed_dim=48, norm_layer=nn.LayerNorm)
    tokens, _, _ = patch_embedding(input_tensor)
    print('shape of tokens:{}'.format(tokens.shape))

    # DeAttention = DeformableAttention(embed_dim=48, n_heads=3, n_head_channels=16,
    #                                   stride=2, norm_layer=nn.BatchNorm2d)
    # tokens = tokens.reshape(2, 64, 64, 48).permute(0, 3, 1, 2)
    # res_attn = DeAttention(tokens)
    # flops, params = profile(DeAttention, inputs=(tokens,))
    # print('Gflops of DeAttention:{}'.format(flops / 10 ** 9))
    # print('params of DeAttention:{}M'.format(params / 10 ** 6))
    # print('shape of attn:{}'.format(res_attn.shape))

    # FFN = MixFeedForward(embed_dim=48, ratio=2)
    # res_ffn = FFN(res_attn)
    # print('shape of ffn:{}'.format(res_ffn.shape))
    #
    # Block = TransformBlock(embed_dim=48, n_heads=3, n_head_channels=16, stride=2, ratio=2, norm_layer=nn.BatchNorm2d)
    # res_block = Block(tokens)
    # print('shape of block:{}'.format(res_block.shape))
    #
    # Down_sample = Downsampling(dim=96, norm_layer=nn.BatchNorm2d)
    # res_down = Down_sample(res_block)
    # print('shape of downsampling:{}'.format(res_down.shape))
    #
    # layer = BasicLayer(embed_dim=48, depth=3, num_heads=3, stride=2)
    # res_layer = layer(tokens)
    # print('shape of basic layer:{}'.format(res_layer.shape))
    #
    # trans = Conv_Former(patch_size=4, in_c=1, embed_dim=96, depths=(3, 3, 4), num_heads=(3, 6, 12))
    # res, feature_map = trans(input_tensor)

    model = ISODformer(mode='non-local', img_size=256, patch_size=4, in_c=1,
                       embed_dim=96, depths=(3, 3, 4), num_heads=(3, 6, 12))
    result = model(input_tensor)
    # print('shape of result:{}'.format(result.shape))

    # model = DATISOD()
    # result = model(input_tensor)
    print('shape of result:{}'.format(result.shape))
