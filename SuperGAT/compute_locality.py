import os
import argparse
import sys
sys.path.insert(0,'/data2/HW/GML/Colab/GML')

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from SuperGAT.data import getattr_d, get_dataset_or_loader
from SuperGAT._tokengt_model import TokenGTNet
from SuperGAT._tokengt_main import get_configuration_string

parser = argparse.ArgumentParser(description='Parser for Supervised Graph Attention Networks')
parser.add_argument("--origin-dir", default="save")
parser.add_argument("--checkpoint-dir", default="checkpoints_")
parser.add_argument('--data-root', default="dataset", metavar='DIR', help='path to dataset')
parser.add_argument("--dataset-class", default="WikiCS")
parser.add_argument("--dataset-name", default="WikiCS")
parser.add_argument('--batch-size', default=128, type=int, metavar='N')
parser.add_argument("--seed", default=42)
parser.add_argument("--data-num-splits", default=1, type=int)

#Training 
parser.add_argument("--model-name", default="TokenGT")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
parser.add_argument("--l2-lambda", default=0.01, type=float)

# Knowledge distillation
parser.add_argument("--use-kd", action="store_true")
parser.add_argument("--w-kd-feat", default = 0.1, type=float)
parser.add_argument("--w-kd-response", default = 0.1, type=float)
parser.add_argument("--freeze-connector", action="store_true")
parser.add_argument("--teacher-name", default="GCN", type=str, choices=["GCN", "GIN", "GAT"])
parser.add_argument("--teacher-lr", default=0.001, type=float)
parser.add_argument("--teacher-l2-lambda", default=0.0, type=float)
parser.add_argument("--teacher-heads", default=8, type=int)
parser.add_argument("--teacher-dropout", default=0.6, type=float)
parser.add_argument("--teacher-num-hidden-features", default=128, type=int)
parser.add_argument("--student-intermediate-index", default=1, type=int)
# For loading teacher model
parser.add_argument("--teacher-checkpoint-dir", default="save/checkpoints_", type=str)
parser.add_argument("--teacher-seed", default=42, type=int)

# TokenGT
parser.add_argument("--input-dropout", type=float, default=0.5)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--attention-dropout", type=float, default=0.0)
parser.add_argument("--activation-dropout", type=float, default=0.5)
parser.add_argument("--classifier-dropout", type=float, default=0.0)
parser.add_argument("--encoder-ffn-embed-dim", type=int, default=128)
parser.add_argument("--encoder-layers", type=int, default=3)
parser.add_argument("--encoder-attention-heads", type=int, default=2)
parser.add_argument("--encoder-embed-dim", type=int, default=128)
parser.add_argument("--broadcast-features", action="store_true")
parser.add_argument("--rand-pe", action="store_true")
parser.add_argument("--rand-pe-dim", type=int)
parser.add_argument("--lap-pe", action="store_true")
parser.add_argument("--lap-pe-k", type=int)
parser.add_argument("--lap-pe-sign-flip", action="store_true")
parser.add_argument("--lap-pe-eig-dropout", type=float)
parser.add_argument("--order-embed", action="store_true", default=True)
parser.add_argument("--coalesce", action="store_true")
parser.add_argument("--performer", action="store_true", default=True)
parser.add_argument("--performer-nb-features", type=int,
                    help="number of random features, defaults to (d * log(d)) where d is head dimension")
parser.add_argument("--performer-feature-redraw-interval", default=0, type=int)
parser.add_argument("--performer-generalized-attention", action="store_true")
parser.add_argument("--performer-no-projection", action="store_true")
parser.add_argument("--activation-fn", choices=('relu', 'gelu'), default='relu')
parser.add_argument("--prenorm", action="store_true", default=True)
parser.add_argument("--postnorm", action="store_true")
parser.add_argument("--drop-edge-tokens", action="store_true")
parser.add_argument("--drop-node-token-features", action="store_true")
args = parser.parse_args()

device = torch.device("cuda:0")

def load_dataset():
    dataset_kwargs = {}
    if args.dataset_class == "ENSPlanetoid":
        dataset_kwargs["neg_sample_ratio"] = args.neg_sample_ratio
    if args.dataset_class == "WikiCS":
        dataset_kwargs["split"] = args.seed % 20  # num_splits = 20
    
    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed, num_splits=args.data_num_splits,
        **dataset_kwargs,
    )
    
    return train_d, val_d, test_d

def get_ckpt_path(args): 
    configuration_str = get_configuration_string(args)
    save_best_path = os.path.join(
        args.origin_dir,
        args.checkpoint_dir,
        args.dataset_name,
        args.model_name,
        configuration_str
    )
    save_best_path = os.path.join(save_best_path, "best.pth")
    print("ckpt path = ", save_best_path)
    return save_best_path
    
    
def load_model(args, train_d): 
    net = TokenGTNet(args, train_d).to(device)
    ckpt_path = get_ckpt_path(args)
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    return net 

def edgeidx_to_adjacency(edge_index, n): 
    # edge_index: [2, e]
    device = edge_index.device
    e = edge_index.shape[1]
    A = torch.zeros(n, n, device=device)
    A[edge_index[0], edge_index[1]] = 1
    return A 

def compute_correlation(x, edge_index):
    # x: [N, d], edge_index: [2, e]
    n, d= x.shape
    e = edge_index.shape[1]
    x_normalized = F.normalize(x, p=2.0, dim=1) # [N, d] 
    A = edgeidx_to_adjacency(edge_index, n) # [N, N]
    
    mask_A = A == 1  # [N, N]
    degree = A.sum(1) # [N]
    zero_pos = degree == 0
    degree[zero_pos] = 1

    out = torch.matmul(x_normalized, x_normalized.transpose(0,1)) # [N, N]
    out = torch.abs(out)
    out[~mask_A] = 0 
    out = out.sum(1) # [N]
    out = torch.div(out, degree) # [N]
    out = out.sum(0) / n # scalar
    return out.item() # scalar

if __name__ == "__main__":
    train_d, val_d, test_d = load_dataset()
    model = load_model(args, train_d)
    model.eval()
    
    if args.rand_pe:
        try:
            rand_pe = torch.load(f"run/{args.dataset_name}/binary-rand-{args.dataset_class}-{args.dataset_name}-{args.rand_pe_dim}.pt")
        except FileNotFoundError:
            assert len(train_d) == 1
            batch = next(iter(train_d))
            n = batch.x.size(0)
            rand_pe = torch.randn(n, args.rand_pe_dim)  # [n_node, D]
            rand_pe = F.normalize(rand_pe, p=2, dim=-1)
            rand_pe[rand_pe >= 0] = 1.
            rand_pe[rand_pe < 0] = -1.
            rand_pe = rand_pe / math.sqrt(args.rand_pe_dim)
        
            torch.save(rand_pe, f"run/{args.dataset_name}/binary-rand-{args.dataset_class}-{args.dataset_name}-{args.rand_pe_dim}.pt")
    else:
        rand_pe = None

    if args.lap_pe:
        try:
            lap_eigvec = torch.load(f"run/{args.dataset_name}/lap-{args.dataset_class}-{args.dataset_name}-{args.lap_pe_k}.pt")
        except FileNotFoundError:
            assert len(train_d) == 1
            batch = next(iter(train_d)) # batch =  Data(edge_index=[2, 182121], test_mask=[18333], train_mask=[18333], val_mask=[18333], x=[18333, 500], y=[18333])
            with torch.no_grad():
                n = batch.x.size(0) # 18333
                batch.edge_index = to_undirected(batch.edge_index, num_nodes=n)
                batch.edge_index, _ = remove_self_loops(batch.edge_index)
                batch.edge_index, _ = add_self_loops(batch.edge_index, num_nodes=n)
                m = batch.edge_index.size(1) # 182121
                np_edge_index = batch.edge_index.detach().cpu().numpy().astype(np.int64)
                row_idx = np_edge_index[0]
                col_idx = np_edge_index[1]
                adj = sp.sparse.coo_matrix((np.ones(m), (row_idx, col_idx)), [n, n])

                in_degree = adj.sum(axis=1)
                in_degree[in_degree == 0] = 1
                normalized_data = 1 / np.multiply(np.sqrt(in_degree[row_idx]), np.sqrt(in_degree[col_idx]))
                normalized_data = np.array(normalized_data).reshape(m)
                normalized_adj = sp.sparse.coo_matrix((normalized_data, (row_idx, col_idx)), [n, n])
                L = sp.sparse.eye(n) - normalized_adj
                eigval, eigvec = sp.sparse.linalg.eigsh(L, k=args.lap_pe_k, which='BE', return_eigenvectors=True)
                lap_eigvec = torch.from_numpy(eigvec).float()  # [N, k]
                lap_eigvec = F.normalize(lap_eigvec, p=2, dim=-1)
                # n =  18333
                # n =  18333
            torch.save(lap_eigvec, f"run/{args.dataset_name}/lap-{args.dataset_class}-{args.dataset_name}-{args.lap_pe_k}.pt")
    else:
        lap_eigvec = None
    
    for batch in train_d: 
        batch = batch.to(device)
        # batch =  Data(edge_index=[2, 431206], test_mask=[11701], train_mask=[11701], val_mask=[11701], x=[11701, 300], y=[11701])
        
        n = batch.x.size(0) # 18333
        batch.edge_index = to_undirected(batch.edge_index, num_nodes=n)
        batch.edge_index, _ = remove_self_loops(batch.edge_index)
        
        # Forward
        if args.lap_pe or args.rand_pe:
            dict_student_features = model(batch.x, batch.edge_index,
                            batch=getattr(batch, "batch", None),
                            attention_edge_index=getattr(batch, "train_edge_index", None),
                            lap_eigvec=lap_eigvec, rand_pe=rand_pe)
        else:
            dict_student_features = model(batch.x, batch.edge_index,
                            batch=getattr(batch, "batch", None),
                            attention_edge_index=getattr(batch, "train_edge_index", None))
        L1_feat = dict_student_features["L1"] # [11701, 128]
        L2_feat = dict_student_features["L2"] # [11701, 128]
        L3_feat = dict_student_features["L3"] # [11701, 128]
        """
        edge_index =  tensor([[    0,     0,     0,  ..., 11700, 11700, 11700],
                              [ 3925,  5830,  7248,  ...,  7586,  8454, 11306]], device='cuda:0') """
        L1_corr = compute_correlation(L1_feat, batch.edge_index)
        L2_corr = compute_correlation(L2_feat, batch.edge_index)
        L3_corr = compute_correlation(L3_feat, batch.edge_index)
        print("L1_corr = ", L1_corr)
        print("L2_corr = ", L2_corr)
        print("L3_corr = ", L3_corr)

    
    
    





