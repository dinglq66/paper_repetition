import torch
from fvcore.nn import FlopCountAnalysis
from thop import profile

from vit_model import Attention


def main():
    # 自注意力机制
    a1 = Attention(dim=512, num_heads=1)
    a1.proj = torch.nn.Identity()

    # 多头注意力机制
    a2 = Attention(dim=512, num_heads=8)

    # [Batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(32, 1024, 512),)

    # flops1 = FlopCountAnalysis(a1, t)
    # print("Self-Attention FLOPs:", flops1.total())
    #
    # flops2 = FlopCountAnalysis(a2, t)
    # print("Multi-Head Attention FLOPs:", flops2.total())

    flop_1, params_1 = profile(a1, inputs=t)
    flop_2, params_2 = profile(a2, inputs=t)
    print('self-attention--flop:{}, params:{}'.format(flop_1, params_1))
    print('multi-head-attention--flop:{}, params:{}'.format(flop_2, params_2))


if __name__ == '__main__':
    main()
