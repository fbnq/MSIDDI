import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import random
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
import os
import time
import math
from utils.data_processing import load_vocab, load_data, save_checkpoint
from utils.metrics import get_scores
from pretraining.MPNN import MPNN


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../model_v7/datachem/DeepDDI_train.csv')
    parser.add_argument('--separate_val_path', type=str, default='../model_v7/datachem/DeepDDI_valid.csv')
    parser.add_argument('--separate_test_path', type=str, default='../model_v7/datachem/DeepDDI_test.csv')
    parser.add_argument('--vocab_path', type=str, default='../model_v7/datachem/drug_list_deep.csv')

    # training
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    # architecture
    parser.add_argument('--num_layer', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Dimensionality of hidden layers in MPNN')
    parser.add_argument('--bias', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--ffn_input', type=int, default=512,
                        help='Dimensionality of input layers in FFN')
    parser.add_argument('--ffn_hidden1', type=int, default=128,
                        help='Dimensionality of input layers in FFN')
    parser.add_argument('--ffn_hidden2', type=int, default=64,
                        help='Dimensionality of input layers in FFN')
    parser.add_argument('--ffn_output', type=int, default=1,
                        help='Dimensionality of input layers in FFN')

    # store
    parser.add_argument('--save_dir', type=str, default='./model_save',
                        help='Directory where model checkpoints will be saved')

    args = parser.parse_args()
    return args

def seed_setting():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7) # for cpu
    torch.cuda.manual_seed_all(7) # for gpu
    torch.backends.cudnn.deterministic = True

def train(args, drug_smiles, edges, labels):
    model = MPNN(args)

    if args.cuda:
        model.cuda()

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    num_train_sample = math.floor(edges.shape[1] * 0.8)
    train_edges = edges[:, : num_train_sample]
    train_labels = labels[: num_train_sample]
    val_edges = edges[:, num_train_sample:]
    val_labels = labels[num_train_sample:]
    print('-----------------Train Model-------------------')
    best_score, best_epoch = 0, 0
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()



        preds = model(drug_smiles, train_edges)
        loss = torch.mean(loss_func(preds, train_labels.float()))


        roc_score, ap_score, f1_score, acc_score = get_scores(model, drug_smiles, val_edges, val_labels)
        if epoch % 10 == 0:
            print('Epoch: {} train_loss= {:.5f} val_roc= {:.5f} val_ap= {:.5f}, val_f1= {:.5f}, val_acc={:.5f}, time= {:.5f}'.format(
                epoch + 1, loss, roc_score, ap_score, f1_score, acc_score, time.time() - t))

        if roc_score > best_score:
            best_score = roc_score
            best_epoch = epoch
            if args.save_dir:
                save_checkpoint(os.path.join(args.save_dir, 'model.pt'), model, args)

        loss.backward()
        if args.vocab_path == 'datachem/drug_list_deep.csv':
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    print('---------------------Optimization Finished!------------------------')


def main():
    args = parse_args()
    seed_setting()

    smiles2idx = load_vocab(args.vocab_path) if args.vocab_path is not None else None
    if smiles2idx is not None:
        idx2smiles = [''] * len(smiles2idx)
        for smiles, smiles_idx in smiles2idx.items():
            idx2smiles[smiles_idx] = smiles
    else:
        idx2smiles = None
    drug_smiles = idx2smiles
    all_edges, all_labels = load_data(args, filepath=args.data_path, smiles2idx=smiles2idx)
    all_edges = torch.from_numpy(all_edges)
    all_labels = torch.from_numpy(all_labels)
    num_nodes = len(smiles2idx)
    print('Number of nodes: {}, Number of edge samples: {}.'.format(num_nodes, all_edges.shape[1]))

    args.cuda = True if torch.cuda.is_available() else False
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        all_edges = all_edges.cuda()
        all_labels = all_labels.cuda()

    train(args, drug_smiles, all_edges, all_labels)

if __name__ == '__main__':
    main()