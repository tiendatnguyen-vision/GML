# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from einops import rearrange

from .multihead_attention import MultiheadAttention
from .performer_pytorch import FastAttention


class MultiheadPerformerAttention(MultiheadAttention):

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            performer_nb_features=None,
            performer_generalized_attention=False,
            performer_no_projection=False,
            attention_dropout=0.0,
            dropout=0.0,
            bias=True,
            self_attention=False
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            attention_dropout,
            dropout,
            bias,
            self_attention,
        )
        assert attention_dropout == 0.0
        self.fast_attention = FastAttention(
            self.head_dim,
            performer_nb_features,
            causal=False,
            generalized_attention=performer_generalized_attention,
            kernel_fn=nn.ReLU(),
            no_projection=performer_no_projection
        )

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

        assert k.size(0) == src_len

        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.num_heads)[None, ...], (q, k, v))
        attn = self.fast_attention(q, k, v)[0]
        attn = rearrange(attn, 'h n d -> n (h d)')

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        return attn
