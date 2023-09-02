from typing import Optional

import einops
import numpy as np
import torch
from hydra.utils import instantiate

from ..components.pose_transformer import TransformerDecoder
from ..components.resnet_decoder import SPADEGenerator_noSPADENorm


def build_texture_head(cfg):
    return instantiate(cfg)

class TextureTransformerHead(TransformerDecoder):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        # token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        # skip_token_embedding: bool = False,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        # Image size must be divisible by patch size
        assert image_size % patch_size == 0
        self.num_tokens_per_dim = (image_size // patch_size)
        self.dim = dim

        super().__init__(
            num_tokens=(self.num_tokens_per_dim ** 2 ),
            token_dim=dim,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            emb_dropout_type=emb_dropout_type,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
            skip_token_embedding=True,
        )

        upsampling_factor = patch_size
        upsampling_factor_log2 = int(np.log2(upsampling_factor))
        assert (2 ** upsampling_factor_log2) == upsampling_factor, "Upsampling factor must be a power of 2"
        self.resnet_head = SPADEGenerator_noSPADENorm(
            img_H=self.image_size,
            img_W=self.image_size,
            nc_init=dim,
            nc_out=2,   # 2 for flow
            n_upconv=upsampling_factor_log2,
            predict_flow=False,
        )

    def forward(self, context):
        context = einops.rearrange(context, 'b c h w -> b (h w) c') 
        inp = context.new_zeros((context.shape[0], self.num_tokens_per_dim ** 2, self.dim))
        out_features = super().forward(inp, context=context)
        out_features = einops.rearrange(out_features, 'b (h w) c -> b c h w', h=self.num_tokens_per_dim, w=self.num_tokens_per_dim) 

        out_texture = self.resnet_head(out_features)
        return out_texture


class TextureTransformerHeadMultiLayerAttn(TextureTransformerHead):
    def forward(self, context_list):
        batch_size = context_list[0].shape[0]
        inp = context_list[0].new_zeros((batch_size, self.num_tokens_per_dim ** 2, self.dim))
        out_features = TransformerDecoder.forward(self, inp, context_list=context_list)
        out_features = einops.rearrange(out_features, 'b (h w) c -> b c h w', h=self.num_tokens_per_dim, w=self.num_tokens_per_dim) 

        out_texture = self.resnet_head(out_features)
        return out_texture
