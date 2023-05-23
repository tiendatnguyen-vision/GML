from typing import Callable, Optional

import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention
from .multihead_performer_attention import MultiheadPerformerAttention
from .feedforward import FeedForward


class TokenGTGraphEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            encoder_layers: int = 12,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            performer: bool = False,
            performer_nb_features: int = None,
            performer_generalized_attention: bool = False,
            performer_no_projection: bool = False,
            activation_fn: str = "relu",
            layernorm_style: str = "postnorm"
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layernorm_style = layernorm_style

        if performer:
            self.self_attn = MultiheadPerformerAttention(
                embedding_dim,
                num_attention_heads,
                performer_nb_features=performer_nb_features,
                performer_generalized_attention=performer_generalized_attention,
                performer_no_projection=performer_no_projection,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
            )
        else:
            self.self_attn = MultiheadAttention(
                embedding_dim,
                num_attention_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
            )
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn=activation_fn,
            activation_dropout=activation_dropout,
            dropout=dropout
        ) if ffn_embedding_dim > 0 else nn.Sequential(
            {'relu': nn.ReLU(inplace=True), 'gelu': nn.GELU()}[activation_fn],
            nn.Dropout(activation_dropout)
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def SelfAttention(self, x, n_nodes, node_output):
        x = self.self_attn(query=x[:n_nodes] if node_output else x, key=x, value=x)
        return x

    def forward(self, x: torch.Tensor, n_nodes: int, node_output: bool):
        residual = x[:n_nodes] if node_output else x
        if self.layernorm_style == "prenorm":
            x = self.self_attn_layer_norm(x)
            x = self.SelfAttention(x=x, n_nodes=n_nodes, node_output=node_output)
            x = residual + x
        elif self.layernorm_style == "postnorm":
            x = self.SelfAttention(x=x, n_nodes=n_nodes, node_output=node_output)
            x = residual + x
            x = self.self_attn_layer_norm(x)
        else:
            raise NotImplementedError

        residual = x
        if self.layernorm_style == "prenorm":
            x = self.final_layer_norm(x)
            x = self.feedforward(x)
            x = residual + x
        elif self.layernorm_style == "postnorm":
            x = self.feedforward(x)
            x = residual + x
            x = self.final_layer_norm(x)
        else:
            raise NotImplementedError
        return x
