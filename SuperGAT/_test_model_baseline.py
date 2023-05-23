from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.conv import GCNConv, GATConv, GINConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import negative_sampling, batched_negative_sampling
import torch_geometric.nn.inits as tgi

from layer import is_pretraining
from layer_cgat import CGATConv
from data import getattr_d, get_dataset_or_loader


def _get_gn_cls(cls_name: str):
    if cls_name == "GAT":
        return GATConv
    elif cls_name == "GCN":
        return GCNConv
    elif cls_name == "GIN":
        return GINConv
    else:
        raise ValueError


def _get_gn_kwargs(cls_name: str, args, channels, **kwargs):
    in_channels, out_channels, hidden_channels = channels
    if cls_name == "GAT":
        return {"in_channels": in_channels,
                "out_channels": out_channels,
                "heads": args.heads,
                "dropout": args.dropout,
                **kwargs}
    elif cls_name == "GCN":
        return {"in_channels": in_channels,
                "out_channels": out_channels}
    elif cls_name == "GIN":
        return {"nn": nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(hidden_channels, out_channels)
        )}
    else:
        raise ValueError


def _get_last_features(cls_name: str, args):
    if cls_name == "GAT":
        return args.num_hidden_features * args.heads
    elif cls_name == "GCN":
        return args.num_hidden_features
    elif cls_name == "SAGE":
        return args.num_hidden_features
    elif cls_name == "GIN":
        return args.num_hidden_features
    else:
        raise ValueError


class _test_MLPNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(_test_MLPNet, self).__init__()
        self.args = args

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.fc = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(num_input_features, args.num_hidden_features),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden_features, args.num_hidden_features),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden_features, args.num_hidden_features),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden_features, num_classes),
        )
        pprint(next(self.modules()))

    def forward(self, x, *args, **kwargs):
        return self.fc(x)


class _test_GNN(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(_test_GNN, self).__init__()
        self.args = args

        gn_layer = _get_gn_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = gn_layer(
            **_get_gn_kwargs(args.model_name, args,
                             (num_input_features, args.num_hidden_features, args.num_hidden_features),
                             concat=True),
        )
        self.conv2 = gn_layer(
            **_get_gn_kwargs(args.model_name, args,
                             (_get_last_features(args.model_name, args), args.num_hidden_features, args.num_hidden_features),
                             concat=True),
        )
        self.conv3 = gn_layer(
            **_get_gn_kwargs(args.model_name, args,
                             (_get_last_features(args.model_name, args), args.num_hidden_features, args.num_hidden_features),
                             concat=True),
        )
        self.conv4 = gn_layer(
            **_get_gn_kwargs(args.model_name, args,
                             (_get_last_features(args.model_name, args), num_classes, args.num_hidden_features),
                             heads=(args.out_heads or args.heads),
                             concat=False),
        )

        self.reset_parameters()
        pprint(next(self.modules()))

    def reset_parameters(self):
        return

    def forward(self, x, edge_index, batch=None, **kwargs):

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv4(x, edge_index)

        return x


if __name__ == '__main__':
    from arguments import get_args

    main_args = get_args(
        model_name="GCN",
        dataset_class="PPI",
        dataset_name="PPI",
        custom_key="NE",
    )

    train_d, val_d, test_d = get_dataset_or_loader(
        main_args.dataset_class, main_args.dataset_name, main_args.data_root,
        batch_size=main_args.batch_size, seed=main_args.seed,
    )

    _m = LinkGNN(main_args, train_d)

    for b in train_d:
        ob = _m(b.x, b.edge_index)
        print(b.x.size(), b.edge_index.size(), ob.size())
