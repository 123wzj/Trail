from argparse import ArgumentParser
from copy import deepcopy
import math
import pickle
from types import SimpleNamespace
import os

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch
from torch.optim import Adam
from tqdm import tqdm

from models.gnn import SageClassifier

# CPU
torch.set_num_threads(16)
HYPERPARAMS = SimpleNamespace(
    icefog_fixed=True,
    ioc_enc=64, hidden=512, layers=3, aggr='max', autoencoder=True, variational=False,
    lr=1e-4, wd=1e-5, epochs=100, bs=16, patience=10, sample_size=512
)
BEST_HP = SimpleNamespace(
    ioc_enc=64, hidden=512, layers=2, aggr='max', autoencoder=True,
    variational=False, lr=0.0001, wd=1e-05, epochs=100, bs=16, patience=10,
    sample_size=32, heads=0
)

def train(hp, model, g, batch_tr, y_tr, batch_va, y_va, batch_te, y_te):
    opt = Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)
    n_batches = math.ceil(batch_tr.size(0)/hp.bs)

    best = {'e': -1, 'va': 0, 'te-acc': 0, 'te-bacc':0, 'sd': None}
    logs = []

    # 第一步：保存原始标签
    original_y = deepcopy((g.event_ids, g.y))

    # Make sure model can only see classes of nodes it's not doing inference on
    # 第二步：为训练集创建标签掩码
    tr_mask = ~((g.event_ids == batch_tr.unsqueeze(-1)).sum(dim=0).bool())
    g_tr_y = g.y[tr_mask]  # 隐藏训练集标签   只保留非训练集的标签
    g_tr_ids = g.event_ids[tr_mask]  # 对应的事件ID

    va_mask = ~((g.event_ids == batch_va.unsqueeze(-1)).sum(dim=0).bool())
    g_va_y = g.y[va_mask]
    g_va_ids = g.event_ids[va_mask]

    te_mask = ~((g.event_ids == batch_te.unsqueeze(-1)).sum(dim=0).bool())
    g_te_y = g.y[te_mask]
    g_te_ids = g.event_ids[te_mask]

    for e in range(hp.epochs):
        perm = torch.randperm(batch_tr.size(0))
        batch_tr = batch_tr[perm]
        y_tr = y_tr[perm]

        prog = tqdm(range(n_batches))

        g.y = g_tr_y
        g.event_ids = g_tr_ids
        for b in prog:
            idx = batch_tr[b*hp.bs : (b+1)*hp.bs]
            y = y_tr[b*hp.bs : (b+1)*hp.bs]

            model.train()
            opt.zero_grad()
            loss = model(g, idx, to_onehot(y))
            loss.backward()
            opt.step()

            prog.desc = f"[{e}] {loss.item():0.2E}"

        model.eval()
        tr_acc, tr_bacc = eval(model, g, batch_va, y_va)
        print(f"\tTr   acc: {tr_acc:0.3f}\tb-acc: {tr_bacc:0.3f}")

        g.y = g_va_y
        g.event_ids = g_va_ids
        va_acc, va_bacc = eval(model, g, batch_va, y_va)
        print(f"\tVal  acc: {va_acc:0.3f}\tb-acc: {va_bacc:0.3f}")

        g.y = g_te_y
        g.event_ids = g_te_ids
        te_acc, te_bacc = eval(model, g, batch_te, y_te)
        print(f"\tTest acc: {te_acc:0.3f}\tb-acc: {te_bacc:0.3f}")

        log = dict(
            epoch=e,
            tr_acc=tr_acc, tr_bacc=tr_bacc,
            va_acc=va_acc, va_bacc=va_bacc,
            te_acc=te_acc, te_bacc=te_bacc,
        )
        logs.append(log)

        if va_bacc > best['va']:
            best['e'] = e
            best['va'] = va_bacc
            best['te-acc'] = te_acc
            best['te-bacc'] = te_bacc
            best['sd'] = deepcopy(model.state_dict())

        if e-best['e'] >= hp.patience:
            print("Early stopping")
            break

    g.event_ids = original_y[0]
    g.y = original_y[1]
    return best, logs

@torch.no_grad()
def eval(model, g, idx, y, print_mat=False):
    model.eval()
    preds = model.inference(g, idx)
    y_hat = preds.argmax(dim=1)

    b_acc = balanced_accuracy_score(y, y_hat)
    acc = accuracy_score(y, y_hat)

    if print_mat:
        print(confusion_matrix(y, y_hat))

    return acc, b_acc

def get_and_split(dataset, folds=5, val=True):
    g = torch.load(f'{dataset}/full_graph_csr.pt', weights_only=False)
    # 5折交叉验证
    kf = StratifiedKFold(n_splits=folds)
    kf_val = StratifiedKFold(n_splits=2)

    # Only eval on OTX so we can better compare results
    valid_mask = (
            g.sources == g.src_map['OTX']  # 来源必须是OTX
    ).logical_and(
        g.y != -1  # 标签必须有效（非未知）
    )

    # Taken care of in graph construction now
    #num_neighbors = torch.tensor([len(n) for n in g.edge_csr[g.event_ids]])
    #valid_mask = num_neighbors != 0

    valid_ids = g.event_ids[valid_mask]
    valid_ys = g.y[valid_mask]

    other_ids = g.y[~valid_mask]  # 其他来源事件
    other_ys = g.y[~valid_mask]  # 其他来源标签

    for tr,te in kf.split(valid_ids, valid_ys):
        if val:
            va_,te_ = next(kf_val.split(te, valid_ys[te]))
            va = te[va_]; te = te[te_]

            tr_idx = torch.cat([valid_ids[tr], other_ids])
            tr_ys = torch.cat([valid_ys[tr], other_ys])

            va_idx = valid_ids[va]
            te_idx = valid_ids[te]

            yield g, \
                (tr_idx, tr_ys), \
                (va_idx, valid_ys[va]), \
                (te_idx, valid_ys[te])
        else:
            tr_idx = torch.cat([valid_ids[tr], other_ids])
            tr_ys = torch.cat([valid_ys[tr], other_ys])

            te_idx = valid_ids[te]
            yield g, \
                (tr_idx, tr_ys), \
                (te_idx, valid_ys[te])

def to_onehot(y, num_classes=24):
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot[torch.arange(y.size(0)), y.long()] = 1.
    return y_onehot


@torch.no_grad()
def get_final_preds(model, g, te_idx, te_y):
    original_y = deepcopy((g.event_ids, g.y))

    # Mask out labels we're testing on
    te_mask = ~((g.event_ids == te_idx.unsqueeze(-1)).sum(dim=0).bool())
    g_te_y = g.y[te_mask]
    g_te_ids = g.event_ids[te_mask]
    g.y = g_te_y
    g.event_ids = g_te_ids

    preds = model.inference(g, te_idx)
    g.event_ids = original_y[0]
    g.y = original_y[1]

    return preds, te_y


def main(hp):
    base_dir = '/root/PythonProject/Trail-main/model_weights'

    # 创建目录
    os.makedirs(f'{base_dir}/weights/{hp.layers}-layer', exist_ok=True)
    os.makedirs(f'{base_dir}/predictions/{hp.layers}-layer', exist_ok=True)
    os.makedirs(f'{base_dir}/logs', exist_ok=True)
    os.makedirs(f'{base_dir}/results/feats', exist_ok=True)

    stats = []
    generator = get_and_split(hp.dataset)
    i = 0

    has_ae = '+ae' if hp.autoencoder else ''
    has_ae += '+var' if hp.variational else ''
    has_ae += f'+attn-{hp.heads}' if hp.heads else ''

    for g, tr, va, te in generator:
        y = tr[1]
        y_onehot = to_onehot(y)

        per_class = y_onehot.sum(dim=0)
        weight = y.size(0) - per_class

        model = SageClassifier(
            hp.dataset, hp.ioc_enc, hp.hidden, (g.y.max() + 1).item(),
            class_weights=weight,
            layers=hp.layers, aggr=hp.aggr, autoencoder=hp.autoencoder,
            sample_size=hp.sample_size, variational=hp.variational,
            heads=hp.heads
        )

        best, log = train(hp, model, g, *tr, *va, *te)
        sd = best.pop('sd')

        model_path = f'{base_dir}/weights/{hp.layers}-layer/gnn_train-{best["va"]:0.3f}_{hp.aggr}_feats{has_ae}-new-data.pt'
        torch.save((sd, model.args, model.kwargs), model_path)
        print(f"模型已保存: {model_path}")

        log_path = f'{base_dir}/logs/feats{has_ae}.pkl'
        with open(log_path, 'wb') as f:
            pickle.dump(log, f)
        print(f"日志已保存: {log_path}")

        stats.append(best)

        model.load_state_dict(sd)
        model.eval()
        preds, ys = get_final_preds(model, g, *te)

        pred_path = f'{base_dir}/predictions/{hp.layers}-layer/feats_gnn-{has_ae}-{i}.pt'
        torch.save((preds, ys), pred_path)
        print(f"预测已保存: {pred_path}")

        i += 1

    # 保存统计结果
    df = pd.DataFrame(stats)
    print(f"\n{'=' * 70}")
    print("训练统计结果:")
    print(f"{'=' * 70}")
    print(df)

    result_path = f'{base_dir}/results/feats/gnn_{hp.layers}-layers{has_ae}.csv'
    with open(result_path, 'a') as f:
        f.write(str(hp) + '\n\n')
        df.to_csv(f)
        f.write('\n均值:\n')
        df.mean().to_csv(f)
        f.write('\n标准误:\n')
        df.sem().to_csv(f)

    print(f"\n结果已保存: {result_path}")
    print(f"\n生成的文件:")
    print(f"  模型权重:  {base_dir}/weights/{hp.layers}-layer/")
    print(f"  预测结果:  {base_dir}/predictions/{hp.layers}-layer/")
    print(f"  训练日志:  {base_dir}/logs/")
    print(f"  统计结果:  {base_dir}/results/feats/")


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--no-ae', action='store_false')
    ap.add_argument('--variational', action='store_true')
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('-s', '--samples', type=int, default=32)
    ap.add_argument('--aggr', default='max')
    ap.add_argument('--attention', type=int, default=0)
    ap.add_argument('--dataset', default='/root/PythonProject/Trail-main/src/otx_dataset')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--ioc_enc', type=int, default=64)
    args = ap.parse_args()

    HYPERPARAMS.autoencoder = args.no_ae
    HYPERPARAMS.variational = args.variational
    HYPERPARAMS.layers = args.layers
    HYPERPARAMS.sample_size = args.samples
    HYPERPARAMS.aggr = args.aggr
    HYPERPARAMS.heads = args.attention
    HYPERPARAMS.dataset = args.dataset
    HYPERPARAMS.hidden = args.hidden
    HYPERPARAMS.ioc_enc = args.ioc_enc
    HYPERPARAMS.dataset = args.dataset
    main(HYPERPARAMS)
