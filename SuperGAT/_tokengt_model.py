import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import random
from data import get_dataset_or_loader, getattr_d

from pprint import pprint
from typing import Tuple, List

import math
import scipy
import numpy as np
from _tokengt_lib import ProjectionUpdater, MultiheadAttention, TokenGTGraphEncoderLayer, gaussian_orthogonal_random_matrix

seed=42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def init_params(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        torch.nn.init.xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        torch.nn.init.xavier_normal_(module.q_proj.weight.data, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_normal_(module.k_proj.weight.data, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_normal_(module.v_proj.weight.data, gain=1 / math.sqrt(2))

class TokenGTNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super().__init__()
        self.args = args

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        assert not (args.prenorm and args.postnorm)
        assert args.prenorm or args.postnorm
        self.encoder_embed_dim = args.encoder_embed_dim
        self.performer = args.performer
        self.broadcast_features = args.broadcast_features
        self.rand_pe = args.rand_pe
        self.rand_pe_dim = args.rand_pe_dim
        self.lap_pe = args.lap_pe
        self.lap_pe_k = args.lap_pe_k
        self.coalesce = args.coalesce
        self.lap_pe_sign_flip = args.lap_pe_sign_flip
        self.order_embed = args.order_embed
        self.drop_edge_tokens = args.drop_edge_tokens  # only for ablation study
        self.drop_node_token_features = args.drop_node_token_features  # only for ablation study

        self.input_dropout = nn.Dropout(args.input_dropout, inplace=True)
        self.node_encoder = nn.Linear(num_input_features, args.encoder_embed_dim)
        if self.broadcast_features:
            self.broadcast_feature_encoder = nn.Linear(2 * num_input_features, args.encoder_embed_dim, bias=False)
        if self.rand_pe:
            self.rand_encoder = nn.Linear(2 * args.rand_pe_dim, args.encoder_embed_dim, bias=False)
        if self.lap_pe:
            self.lap_encoder = nn.Linear(2 * args.lap_pe_k, args.encoder_embed_dim, bias=False)
            self.lap_eig_dropout = nn.Dropout(p=args.lap_pe_eig_dropout) if args.lap_pe_eig_dropout > 0 else None
        if self.order_embed:
            self.order_encoder = nn.Embedding(2, args.encoder_embed_dim)

        self.layers = nn.ModuleList(
            [
                TokenGTGraphEncoderLayer(
                    embedding_dim=args.encoder_embed_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    encoder_layers=args.encoder_layers,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    performer=args.performer,
                    performer_nb_features=args.performer_nb_features,
                    performer_generalized_attention=args.performer_generalized_attention,
                    performer_no_projection=args.performer_no_projection,
                    activation_fn=args.activation_fn,
                    layernorm_style="prenorm" if args.prenorm else "postnorm"
                )
                for layer_idx in range(args.encoder_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(args.encoder_embed_dim) if args.prenorm else None
        self.classifier_dropout = nn.Dropout(args.classifier_dropout, inplace=True)
        self.embed_out = nn.Linear(args.encoder_embed_dim, num_classes)
        
        ### Knowledge distillation 
        if args.teacher_name == "GAT":
            self.connector = nn.Linear(args.encoder_embed_dim, args.teacher_num_hidden_features * args.teacher_heads)
        else: 
            self.connector = nn.Linear(args.encoder_embed_dim, args.teacher_num_hidden_features)

        if args.freeze_connector:
            for param in self.connector.parameters():
                param.requires_grad = False
        
        self.apply(init_params)

        if args.performer:
            self.performer_proj_updater = ProjectionUpdater(self.layers, args.performer_feature_redraw_interval)

        pprint(next(self.modules()))

    @staticmethod
    def get_random_sign_flip(eigvec):
        sign_flip = torch.rand(1, eigvec.size(1), device=eigvec.device, dtype=eigvec.dtype)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        return sign_flip

    @staticmethod
    def get_index_embed(pe, edge_index):
        # pe: [18333, 500]
        node_pe = torch.cat((pe, pe), dim=-1)  # [node_num, 2D]
        edge_pe = torch.cat((pe[edge_index[0]], pe[edge_index[1]]), dim=-1)  # [edge_num, 2D]
        return torch.cat((node_pe, edge_pe), dim=0)  # [node_num + edge_num, 2D]

    def get_order_embed(self, node_num, edge_index):
        device = edge_index.device
        order = torch.cat((
            torch.ones(node_num, device=device, dtype=torch.long),
            torch.eq(edge_index[0], edge_index[1]).long()
        ))
        return self.order_encoder(order)  # [node_num + edge_num, D]

    def tokenize(self, x, lap_eigvec, rand_pe, edge_index):
        # x: [18333, 500]
        node_num = x.size(0)           # 18333
        edge_num = edge_index.size(1)  # 163788
        dtype = x.dtype
        device = x.device
        full_feature = torch.zeros(node_num + edge_num, self.encoder_embed_dim, device=device, dtype=dtype) # [182121, 256]
        
        if self.broadcast_features:
            feature_index_embed = self.get_index_embed(x, edge_index)  # [n_node + n_edge, 2D] = [182121, 1000]
            full_feature = full_feature + self.broadcast_feature_encoder(feature_index_embed) # [182121, 256]
            if self.drop_node_token_features:
                full_feature[:node_num] = 0.
        else:
            node_feature = self.node_encoder(x)  # [n_node, D]
            full_feature[:node_num].copy_(node_feature)

        if self.rand_pe:
            assert rand_pe.size(0) == node_num
            rand_pe = rand_pe.to(device) # [18333, 256]
            rand_index_embed = self.get_index_embed(rand_pe, edge_index)  # [n_node + n_edge, 2D] = [182121, 512]
            full_feature = full_feature + self.rand_encoder(rand_index_embed) # [182121, 256]
            
        if self.lap_pe:
            assert lap_eigvec.size(0) == node_num
            lap_eigvec = lap_eigvec.to(device)
            if self.lap_eig_dropout is not None:
                lap_eigvec = self.lap_eig_dropout(lap_eigvec)
            if self.lap_pe_sign_flip and self.training:
                lap_eigvec = lap_eigvec * self.get_random_sign_flip(lap_eigvec)
            lap_index_embed = self.get_index_embed(lap_eigvec, edge_index)  # [n_node + n_edge, 2Dl]
            full_feature = full_feature + self.lap_encoder(lap_index_embed)

        if self.order_embed:
            full_feature = full_feature + self.get_order_embed(node_num, edge_index)

        return full_feature  # [T, D]

    def performer_fix_projection_matrices_(self):
        self.performer_proj_updater.feature_redraw_interval = None

    def forward(self, x, edge_index, batch=None, lap_eigvec=None, rand_pe=None, **kwargs):
        # x is node features
        if self.performer:
            self.performer_proj_updater.redraw_projections()
        # x: [18333, 500]
        # edge_index: [2, 163788]
        # lap_eigvec: [18333, 64]
          
        n_nodes = x.size(0)
        x = self.input_dropout(x) # !!
        x = self.tokenize(x, lap_eigvec, rand_pe, edge_index)  # x: T x C
        
        # x: [182121, 256]
        dict_features = {}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x, n_nodes=n_nodes, node_output=(i == len(self.layers) - 1) or self.drop_edge_tokens)
            if self.coalesce and (not i == len(self.layers) - 1):
                x[:n_nodes] = x[:n_nodes] + \
                              torch.sparse_coo_tensor(edge_index[0:1], x[n_nodes:],
                                                      size=(n_nodes, x.size(-1))).coalesce().to_dense()
            dict_features["L{}".format(i+1)] = x 

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
    
        x = self.embed_out(self.classifier_dropout(x))
        dict_features["logits"] = x 
        return dict_features
