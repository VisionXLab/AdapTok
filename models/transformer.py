import torch
import torch.nn as nn
from .block import Block as BlockTimm
# from timm.models.vision_transformer import Block as BlockTimm

from models import register


@register('transformer_encoder_fused')
class TransformerEncoderFused(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        assert ff_dim is None
        assert dim == head_dim * n_head

        self.blocks = nn.Sequential(
            *[
                BlockTimm(
                    dim=dim,
                    num_heads=n_head,
                    mlp_ratio=4,
                    qkv_bias=False,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


@register('transformer_encoder_parallel')
class TransformerEncoderParallel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        n_head,
        head_dim,
        ff_dim=None,
        dropout=0.0
    ):
        super().__init__()
        self.is_encoder_decoder = True
        assert ff_dim is None
        assert dim == head_dim * n_head
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BlockTimm(
                    dim=dim,
                    num_heads=n_head,
                    mlp_ratio=4,
                    qkv_bias=False,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
            )

    def forward(self, context, query, attn_mask=None):
        query_length = query.size(1)
        h = torch.cat([context, query], dim=1)

        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)

        h = h[:, -query_length:, :]
        return h


@register('transformer_scorer')
class TransformeScorer(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, latent_len, step, ff_dim=None, dropout=0.0):
        super().__init__()

        self.step = step

        self.layers = nn.ModuleList()
        assert ff_dim is None
        assert dim == head_dim * n_head
        self.project = nn.Linear(dim * 2, dim)
        self.latent_pe = nn.Parameter((dim ** -0.5) * torch.randn(latent_len, dim), requires_grad=True)

        self.blocks = nn.Sequential(
            *[
                BlockTimm(
                    dim=dim,
                    num_heads=n_head,
                    mlp_ratio=4,
                    qkv_bias=False,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.ln_post = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.Tanh(),
            nn.Linear(2 * dim, 1),
        )

    def forward(self, input, quant, attn_mask=None):
        x = self.project(torch.cat([input, quant], dim =-1))
        batchsize, seq_len, _ = x.shape
        x = x + self.latent_pe[:seq_len]
        
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_post(x)

        x = self.ffn(x).squeeze(-1)

        x = x[:, (self.step-1)::self.step]

        return x