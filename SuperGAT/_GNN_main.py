import os
import random
from collections import deque, defaultdict
from typing import Tuple, Any, List, Dict
from copy import deepcopy
from pprint import pprint
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy as sp
import numpy as np
import pandas as pd
from termcolor import cprint
from tqdm import tqdm
from sklearn.metrics import f1_score
import sys
sys.path.insert(0,'/data2/HW/GML/Colab/GML')

from SuperGAT._GNN_arguments import get_important_args, save_args, get_args, pprint_args, get_args_key
from SuperGAT._test_model_baseline import _test_MLPNet, _test_GNN

from SuperGAT.data import getattr_d, get_dataset_or_loader
from SuperGAT.layer import SuperGAT
from SuperGAT.utils import create_hash, to_one_hot, get_accuracy, cprint_multi_lines, blind_other_gpus

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import math
from torch.utils.tensorboard import SummaryWriter

def get_configuration_string(_args): 
    configuration_str = "InputDrop{}_lr{}_hidden-D{}_l2lambda{}".format(_args.dropout, _args.lr, _args.num_hidden_features, _args.l2_lambda)
    return configuration_str

def get_model_path(target_epoch, _args, **kwargs):
    args_key = get_args_key(_args)
    configuration_str = get_configuration_string(_args)

    dir_path = os.path.join(
        _args.checkpoint_dir, 
        _args.dataset_name, 
        _args.model_name,
        configuration_str
    )

    if target_epoch is not None:  # If you want to load the model of specific epoch.
        return os.path.join(dir_path, "{}.pth".format(str(target_epoch).rjust(7, "0")))
    else:
        files_in_checkpoints = [f for f in os.listdir(dir_path) if f.endswith(".pth")]
        if len(files_in_checkpoints) > 0:
            latest_file = sorted(files_in_checkpoints)[-1]
            return os.path.join(dir_path, latest_file)
        else:
            raise FileNotFoundError("There should be saved files in {} if target_epoch is None".format(
                os.path.join(_args.checkpoint_dir, args_key),
            ))

def get_logdir_path(_args): 
    configuration_str = get_configuration_string(_args)
    logdir = os.path.join("save/logs", _args.dataset_name, _args.model_name, configuration_str)
    return logdir

def save_model(model, _args, target_epoch, perf, **kwargs) -> bool:
    try:
        full_path = get_model_path(target_epoch, _args, **kwargs)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(
            obj={
                'model_state_dict': model.state_dict(),
                'epoch': target_epoch,
                'perf': perf,
                **kwargs,
            },
            f=full_path,
        )
        save_args(os.path.dirname(full_path), _args)
        cprint("Save {}".format(full_path), "green")
        return True
    except Exception as e:
        cprint("Cannot save model, {}".format(e), "red")
        return False


def load_model(model, _args, target_epoch=None, **kwargs) -> Tuple[Any, dict] or None:
    try:
        full_path = get_model_path(target_epoch, _args, **kwargs)
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        cprint("Load {}".format(full_path), "green")
        return model, {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    except Exception as e:
        cprint("Cannot load model, {}".format(e), "red")
        return None


def train_model(device, model, dataset_or_loader, criterion, optimizer, epoch, _args):
    model.train()
    try:
        dataset_or_loader.train()
    except AttributeError:
        pass

    total_loss = 0.
    for batch in dataset_or_loader:
        batch = batch.to(device)
        # batch =  Data(edge_index=[2, 163788], test_mask=[18333], train_mask=[18333], val_mask=[18333], x=[18333, 500], y=[18333])
        # batch.train_mask =  tensor([True, True, True,  ..., True, True, True], device='cuda:0')
        # batch.val_mask =  tensor([False, False, False,  ..., False, False, False], device='cuda:0')
        n = batch.x.size(0) # 18333
        batch.edge_index = to_undirected(batch.edge_index, num_nodes=n)
        batch.edge_index, _ = remove_self_loops(batch.edge_index)
        if not _args.model_name == "TokenGT":
            batch.edge_index, _ = add_self_loops(batch.edge_index, num_nodes=n)

        optimizer.zero_grad()
        
        # Forward
        dict_features = model(batch.x, batch.edge_index, batch=getattr(batch, "batch", None))
        outputs = dict_features["logits"]
        """
        # outputs: [N, C] where N is number of nodes and C is number of classes 
        outputs =   tensor([[ 0.5493, -0.4239,  0.0191,  ...,  0.0954,  0.0425,  0.0427],
                            [ 0.3873, -0.2640, -0.1928,  ..., -0.4189, -0.0803, -0.2305],
                            ...,
                            [ 0.3950, -0.0309,  0.0896,  ...,  0.0752, -0.3386, -0.2932]])
        """
        # Loss
        if "train_mask" in batch.__dict__:
            loss = criterion(outputs[batch.train_mask], batch.y[batch.train_mask])
        else:
            loss = criterion(outputs, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def test_model(device, model, dataset_or_loader, criterion, _args, val_or_test="val", verbose=0, **kwargs):
    model.eval()
    try:
        model.set_layer_attrs("cache_attention", _args.task_type == "Attention_Dist")
    except AttributeError:
        pass
    try:
        dataset_or_loader.eval()
    except AttributeError:
        pass

    num_classes = getattr_d(dataset_or_loader, "num_classes")

    total_loss = 0.
    outputs_list, ys_list, batch = [], [], None
    with torch.no_grad():
        for batch in dataset_or_loader:
            batch = batch.to(device)
            n = batch.x.size(0)
            batch.edge_index = to_undirected(batch.edge_index, num_nodes=n)
            batch.edge_index, _ = remove_self_loops(batch.edge_index)
            if not _args.model_name == "TokenGT":
                batch.edge_index, _ = add_self_loops(batch.edge_index, num_nodes=n)

            # Forward
            dict_features = model(batch.x, batch.edge_index, batch=getattr(batch, "batch", None))
            outputs = dict_features["logits"]
            
            # Loss
            if "train_mask" in batch.__dict__:
                val_or_test_mask = batch.val_mask if val_or_test == "val" else batch.test_mask
                loss = criterion(outputs[val_or_test_mask], batch.y[val_or_test_mask]) # !!!!!!!!!!!!!!!!!
                outputs_ndarray = outputs[val_or_test_mask].cpu().numpy()
                ys_ndarray = to_one_hot(batch.y[val_or_test_mask], num_classes)
            elif _args.dataset_name == "PPI":  # PPI task
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), batch.y.cpu().numpy()
            else:
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), to_one_hot(batch.y, num_classes)
            total_loss += loss.item()

            outputs_list.append(outputs_ndarray)
            ys_list.append(ys_ndarray)

    outputs_total, ys_total = np.concatenate(outputs_list), np.concatenate(ys_list)

    if _args.task_type == "Link_Prediction":
        if "run_link_prediction" in kwargs and kwargs["run_link_prediction"]:
            val_or_test_edge_y = batch.val_edge_y if val_or_test == "val" else batch.test_edge_y
            layer_idx_for_lp = kwargs["layer_idx_for_link_prediction"] \
                if "layer_idx_for_link_prediction" in kwargs else -1
            perfs = SuperGAT.get_link_pred_perfs_by_attention(model=model, edge_y=val_or_test_edge_y,
                                                              layer_idx=layer_idx_for_lp)
        else:
            perfs = get_accuracy(outputs_total, ys_total)
    elif _args.perf_type == "micro-f1" and _args.dataset_name == "PPI":
        preds = (outputs_total > 0).astype(int)
        perfs = f1_score(ys_total, preds, average="micro") if preds.sum() > 0 else 0
    elif _args.perf_type == "accuracy" or _args.task_type == "Attention_Dist":
        perfs = get_accuracy(outputs_total, ys_total)
    else:
        raise ValueError

    if verbose >= 2:
        full_name = "Validation" if val_or_test == "val" else "Test"
        cprint("\n[{} of {}]".format(full_name, model.__class__.__name__), "yellow")
        cprint("\t- Perfs: {}".format(perfs), "yellow")

    return perfs, total_loss


def save_loss_and_perf_plot(list_of_list, return_dict, args, columns=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    os.makedirs("plots", exist_ok=True)
    sns.set(style="whitegrid")
    sz = len(list_of_list[0])
    columns = columns or ["col_{}".format(i) for i in range(sz)]
    df = pd.DataFrame(np.transpose(np.asarray([*list_of_list])), list(range(sz)), columns=columns)

    print("\t".join(["epoch"] + list(str(r) for r in range(sz))))
    for col_name, row in zip(df, df.values.transpose()):
        print("\t".join([col_name] + [str(round(r, 5)) for r in row]))
    cprint_multi_lines("\t- ", "yellow", **return_dict)

    plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
    title = "{}-{}-{}".format(args.model_name, args.dataset_name, args.custom_key)
    plot.set_title(title)
    plot.get_figure().savefig("plots/{}_{}_{}.png".format(title, args.seed, return_dict["best_test_perf_at_best_val"]))
    plt.clf()


def _get_model_cls(model_name: str):
    if model_name in ("GCN", "GIN", "GAT"):
        return _test_GNN
    elif model_name == "MLP":
        return _test_MLPNet
    else:
        raise ValueError


def run(args, gpu_id=None, return_model=False, return_time_series=False):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    
    log_dir = get_logdir_path(args)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_perf = 0.
    test_perf_at_best_val = 0.
    best_test_perf = 0.
    best_test_perf_at_best_val = 0.
    link_test_perf_at_best_val = 0.

    val_loss_deque = deque(maxlen=args.early_stop_queue_length)
    val_perf_deque = deque(maxlen=args.early_stop_queue_length)

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
    # train_d =  MyCoauthorCS(), val_d = test_d = none

    net_cls = _get_model_cls(args.model_name)
    net = net_cls(args.model_name, args.heads, args.dropout, args.num_hidden_features, train_d)
    net = net.to(running_device)
    
    params = sum([np.prod(p.size()) for p in net.parameters()])
    print(params)
    
    if args.continue_training: 
        loaded = load_model(net, args, target_epoch=None)
    else: 
        loaded = None

    if loaded is not None:
        net, other_state_dict = loaded
        best_val_perf = other_state_dict["perf"]
        args.start_epoch = other_state_dict["epoch"]

    loss_func = eval(str(args.loss)) or nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    adam_optim = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_lambda)

    ret = {}
    val_perf_list, test_perf_list, val_loss_list = [], [], []
    perf_task_for_val = getattr(args, "perf_task_for_val", "Node")
    best_val_loss = 100
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
        train_loss = train_model(running_device, net, train_d, loss_func, adam_optim, epoch=epoch, _args=args)

        if args.verbose >= 2 and epoch % args.val_interval == 0:
            print("\n\t- Train loss: {}".format(train_loss))

        # Validation.
        if epoch % args.val_interval == 0:
            val_perf, val_loss = test_model(running_device, net, val_d or train_d, loss_func,
                                            _args=args, val_or_test="val", verbose=args.verbose,
                                            run_link_prediction=(perf_task_for_val == "Link"))
            test_perf, test_loss = test_model(running_device, net, test_d or train_d, loss_func,
                                              _args=args, val_or_test="test", verbose=0,
                                              run_link_prediction=(perf_task_for_val == "Link"))
            if args.save_plot:
                val_perf_list.append(val_perf)
                test_perf_list.append(test_perf)
                val_loss_list.append(val_loss)

            if test_perf > best_test_perf:
                best_test_perf = test_perf

            if val_perf >= best_val_perf:

                print_color = "yellow"
                best_val_perf = val_perf
                test_perf_at_best_val = test_perf

                if test_perf_at_best_val > best_test_perf_at_best_val:
                    best_test_perf_at_best_val = test_perf_at_best_val

                if args.task_type == "Link_Prediction":
                    link_test_perf, _ = test_model(running_device, net, test_d or train_d, loss_func,
                                                   _args=args, val_or_test="test", verbose=0,
                                                   run_link_prediction=True)
                    link_test_perf_at_best_val = link_test_perf

            else:
                print_color = None

            ret = {
                "best_val_perf": best_val_perf,
                "test_perf_at_best_val": test_perf_at_best_val,
                "best_test_perf": best_test_perf,
                "best_test_perf_at_best_val": best_test_perf_at_best_val,
            }
            if args.verbose >= 1:
                cprint_multi_lines("\t- ", print_color, **ret)

            # Check early stop condition
            if args.use_early_stop and current_iter > args.early_stop_patience:
                recent_val_loss_mean = float(np.mean(val_loss_deque))
                val_loss_change = abs(recent_val_loss_mean - val_loss) / recent_val_loss_mean
                recent_val_perf_mean = float(np.mean(val_perf_deque))
                val_perf_change = abs(recent_val_perf_mean - val_perf) / recent_val_perf_mean

                if (val_loss_change < args.early_stop_threshold_loss) or \
                        (val_perf_change < args.early_stop_threshold_perf):
                    if args.verbose >= 1:
                        cprint("Early Stopped at epoch {}".format(epoch), "red")
                        cprint("\t- val_loss_change is {} (thres: {}) | {} -> {}".format(
                            round(val_loss_change, 6), round(args.early_stop_threshold_loss, 6),
                            recent_val_loss_mean, val_loss,
                        ), "red")
                        cprint("\t- val_perf_change is {} (thres: {}) | {} -> {}".format(
                            round(val_perf_change, 6), round(args.early_stop_threshold_perf, 6),
                            recent_val_perf_mean, val_perf,
                        ), "red")
                    break
            val_loss_deque.append(val_loss)
            val_perf_deque.append(val_perf)
            
            # log
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            
            writer.add_scalar("Perf/val", val_perf, epoch)
            writer.add_scalar("Perf/test", test_perf, epoch)
            
        print("best_val_loss = ", best_val_loss)
        if args.save_model:
            if args.save_last_only:
                if epoch == args.start_epoch + args.epochs:
                    save_model(net, args, target_epoch=epoch, perf=val_perf)
            else: 
                if epoch % args.save_ckpt_interval == 0: 
                    save_model(net, args, target_epoch=epoch, perf=val_perf)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                configuration_str = get_configuration_string(args)
                save_best_path = os.path.join(
                    args.checkpoint_dir,
                    args.dataset_name,
                    args.model_name,
                    configuration_str
                )
                save_best_path = os.path.join(save_best_path, "best.pth")
                torch.save(
                    obj={'model_state_dict': net.state_dict(), 'epoch': epoch, 'perf': val_perf},
                    f=save_best_path)
            
    if args.task_type == "Link_Prediction":
        ret = {"link_test_perf_at_best_val": link_test_perf_at_best_val, **ret}

    if args.save_plot:
        save_loss_and_perf_plot([val_loss_list, val_perf_list, test_perf_list], ret, args,
                                columns=["val_loss", "val_perf", "test_perf"])
    # save output statistics
    configuration_str = get_configuration_string(args)
    outf_save_dir = os.path.join(
        args.outf_dir,
        args.dataset_name,
        args.model_name,
        configuration_str
    )
    os.makedirs(outf_save_dir, exist_ok=True)
    outf_save_path = os.path.join(outf_save_dir, "outf.json")
    json_outf_obj = json.dumps(ret, indent=4)
    with open(outf_save_path, "w") as outfile:
        outfile.write(json_outf_obj)
        
    if return_model:
        return net, ret
    if return_time_series:
        return {"val_loss_list": val_loss_list, "val_perf_list": val_perf_list, "test_perf_list": test_perf_list, **ret}

    return ret


def run_with_many_seeds(args, num_seeds, gpu_id=None, **kwargs):
    results = defaultdict(list)
    for i in range(num_seeds):
        cprint("## TRIAL {} ##".format(i), "yellow")
        _args = deepcopy(args)
        _args.seed = _args.seed + i
        ret = run(_args, gpu_id=gpu_id, **kwargs)
        for rk, rv in ret.items():
            results[rk].append(rv)
    return results

def summary_results(results_dict: Dict[str, list or float], num_digits=3, keys_to_print=None):
    line_list = []

    def cprint_and_append(x, color=None):
        cprint(x, color)
        line_list.append(x)

    cprint_and_append("## RESULTS SUMMARY ##", "yellow")
    is_value_list = False
    for rk, rv in sorted(results_dict.items()):
        if keys_to_print is not None and rk not in keys_to_print:
            continue
        if isinstance(rv, list):
            cprint_and_append("{}: {} +- {}".format(
                rk, round(float(np.mean(rv)), num_digits), round(float(np.std(rv)), num_digits))
            )
            is_value_list = True
        else:
            cprint_and_append("{}: {}".format(rk, rv))
    cprint_and_append("## RESULTS DETAILS ##", "yellow")
    if is_value_list:
        for rk, rv in sorted(results_dict.items()):
            if keys_to_print is not None and rk not in keys_to_print:
                continue
            cprint_and_append("{}: {}".format(rk, rv))
    return line_list


if __name__ == '__main__':
    main_args = get_args(
        model_name="GCN",  # GAT, GCN
        dataset_class="Planetoid",  # Planetoid, FullPlanetoid, RandomPartitionGraph
        dataset_name="Cora",  # Cora, CiteSeer, PubMed, rpg-10-500-0.1-0.025
        custom_key="EV13NSO8",  # NEO8, NEDPO8, EV13NSO8, EV9NSO8, EV1O8, EV2O8, -500, -Link, -ES, -ATT
    )
    pprint_args(main_args)

    # noinspection PyTypeChecker
    many_seeds_result = run_with_many_seeds(main_args, 1, gpu_id = main_args.gpu_id)

    pprint_args(main_args)
    summary_results(many_seeds_result)
