import pandas as pd
import os
import numpy as np
import torch

def load_vocab(filepath):
    df = pd.read_csv(filepath, index_col=False)
    smiles2id = {smiles: idx for smiles, idx in zip(df['smiles'], range(len(df)))}
    return smiles2id

def load_csv_data(filepath, smiles2id):
    df = pd.read_csv(filepath, index_col=False)
    edges = []  # 数据集中一对节点相连
    labels = []
    # count = 0
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['smiles_1']
        smiles_2 = row_dict['smiles_2']
        if smiles_1 in smiles2id.keys() and smiles_2 in smiles2id.keys():
            idx_1 = smiles2id[smiles_1]
            idx_2 = smiles2id[smiles_2]
            label = int(row_dict['label'])
            edges.append((idx_1, idx_2))
            labels.append(label)
            edges.append((idx_2, idx_1))
            labels.append(label)
        else:
            continue
    edges = np.array(edges, dtype=np.int)
    return edges, labels

def load_data(args, filepath, smiles2idx = None):

    ext = os.path.splitext(filepath)[-1]
    # 将文件名和扩展名分开
    if args.separate_val_path is not None and args.separate_test_path is not None and ext == '.csv':
        assert smiles2idx is not None

        train_edges, train_labels = load_csv_data(filepath, smiles2idx)
        # 返回训练数据集中的edges，train_edges是相连的节点对，train_edges_false是不相连的节点对
        val_edges, val_labels = load_csv_data(args.separate_val_path, smiles2idx)
        test_edges, test_labels = load_csv_data(args.separate_test_path, smiles2idx)

        all_edges = np.concatenate([train_edges, val_edges, test_edges], axis=0)
        all_labels = np.concatenate([train_labels, val_labels, test_labels], axis = 0)

        return np.transpose(all_edges), all_labels

def save_checkpoint(path, model, args):
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': None,
        'features_scaler': None
    }
    torch.save(state, path)
