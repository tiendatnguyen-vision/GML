import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange


class MultiheadAttention(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            attention_dropout=0.0,
            dropout=0.0,
            bias=True,
            self_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.attention_dropout_module = nn.Dropout(attention_dropout, inplace=True)
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout, inplace=True)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor]) -> Tensor:
        tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, embed_dim]
        if key is not None:
            src_len, _ = key.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.num_heads), (q, k, v))

        attn_weights = torch.einsum('hnd,hmd->hnm', q, k).softmax(-1)
        attn_probs = self.attention_dropout_module(attn_weights)
        attn = torch.einsum('hnm,hmd->hnd', attn_probs, v)
        attn = rearrange(attn, 'h n d -> n (h d)')

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        return attn
