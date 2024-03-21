# Copyright (c) 2023, Albert Gu, Tri Dao.


from functools import partial

import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None



class MultiLayerMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int = 2,
        rms_norm: bool = False
    ):
        # init
        super().__init__()

        def create_block():
            norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5)
            block = Block(d_model, Mamba, norm_cls=norm_cls)
            return block

        self.layers = nn.ModuleList([create_block() for _ in range(n_layer)])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=1e-5)

    def forward(self, x, inference_params=None):
        # x: [batch, length, dim]
        residual = None
        for layer in self.layers:
            x, residual = layer(
                x, residual, inference_params=inference_params
            )
        return x