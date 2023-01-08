import time
import math
import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import _pickle as pickle
from model_seq import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=823, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--wd',type=float, default=0, help='Weight decay.')
parser.add_argument('--layers', type=int, default=8, help='Number of hidden layers.')
parser.add_argument('--units', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.7, help='alpha')
parser.add_argument('--lamda', type=float, default=1.5, help='lamda')
parser.add_argument('--variant', action='store_true', default=False, help='GCN model.')
parser.add_argument('--cutoff', type=float, default=14, help='Distance map cutoff.')
parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold.')
parser.add_argument('--pw', type=float, default=1.0, help='Positive weight.')
parser.add_argument('--lw', type=float, default=1.0, help='Peptide loss weight')
args = parser.parse_args()
layers = args.layers
units = args.units
dropout = args.dropout
lamda=args.lamda
alpha=args.alpha
variant=args.variant
cutoff = args.cutoff
threshold = args.threshold
cudaid = "cuda:" + str(args.dev)
device = torch.device(cudaid)
pos_weight = torch.tensor(args.pw)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# Normalize adjacency matrix
def normalize_adjacency_matrix(mat):
    rowsum = np.array(mat.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mat_norm = r_mat_inv @ mat @ r_mat_inv
    return mat_norm

# Get adjacency matrix
def get_adjacency_matrix(ID, cutoff, pro_length):
    dis_mat = np.load('../data/train_matrix/dismat/' + ID + '_dismat.npy')
    adj_mat = dis_mat
    for i in range(len(dis_mat)):
        for j in range(len(dis_mat)):
            if i == j: adj_mat[i][j] = 0
            elif dis_mat[i][j] <= cutoff: adj_mat[i][j] = 1
            else: adj_mat[i][j] = 0
    norm_adj_mat = normalize_adjacency_matrix(adj_mat[:pro_length, :pro_length])
    np.save('../data/train_matrix/adjmat/'+ ID +'_adjmat.npy', norm_adj_mat)
    return norm_adj_mat


def train():
    model.train()
    loss_train = 0
    pairs_number = 0
    for batch in range(0, len(fea)):
        pairs_number += 1
        node_fea = fea[batch]
        node_fea = torch.FloatTensor(node_fea).to(device)
        pro_length = pro_len_list[batch]
        adj_mat_pro = get_adjacency_matrix(dic[batch], args.cutoff, pro_length)
        adj_mat_pro = torch.FloatTensor(adj_mat_pro).to(device)
        label = lab[batch]
        label = torch.FloatTensor(label).to(device)

        optimizer.zero_grad()
        node_fea_pro, node_fea_pep = node_fea[:pro_length], node_fea[pro_length:]
        output = model(node_fea_pro, adj_mat_pro, node_fea_pep)
        label = label.view(-1,1)
        loss_tra_pro = loss_fcn(output[:pro_length], label[:pro_length])
        loss_tra_pep = loss_fcn(output[pro_length:], label[pro_length:])
        loss_tra = loss_tra_pro + args.lw * loss_tra_pep

        loss_tra.backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, GRAD_CLIP)
        optimizer.step()

        loss_train += loss_tra.item()
    print(pairs_number)
    loss_counter = fea.shape[0]
    loss_train /= loss_counter
    return loss_train



if __name__ == '__main__':
    SetName = 'trainset'
    print('Load '+SetName+'...')
    with open('../data/settrain_fea.pkl','rb') as f:
        fea = pickle.load(f)
    with open('../data/settrain_lab.pkl','rb') as f:
        lab = pickle.load(f)
    dic = np.load('../data/settrain_dict.npy')
    pro_len_list = np.load('../data/settrain_prolen.npy')
    pep_len_list = np.load('../data/settrain_peplen.npy')
    print('Load done.')

    # disrupting data
    pair_num = fea.shape[0]
    random_index = torch.randperm(pair_num)
    fea, lab, dic = fea[random_index], lab[random_index], dic[random_index]
    pro_len_list, pep_len_list = pro_len_list[random_index], pep_len_list[random_index] 


    GRAD_CLIP = 5.  # Clipping Gradient
    loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = deepGCN(nlayers=args.layers,
                nfeat=fea[0][0].shape[0],
                nhidden=args.units,
                nclass=1,
                dropout=args.dropout,
                lamda=args.lamda,
                alpha=args.alpha,
                variant=args.variant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    t1 = time.time()
    for epoch in range(0, args.epochs):
        loss_train = train()
        t2 = time.time()
        time_cost = t2 - t1
        t1 = t2
        print('Epoch:'+"%d"%(epoch+1)+'\tTime_cost:'+"%.4f"%time_cost+'\tTra_loss:'+"%.5f"%loss_train)
    torch.save(model.state_dict(), os.path.join("../model", 'modelseq%.1f_%.1f_%d_%d_%d.dat'%(args.pw, args.lw, args.epochs, args.units, args.layers)))
