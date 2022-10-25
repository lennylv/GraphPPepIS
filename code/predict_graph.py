import numpy as np
import random
import math
import torch
import _pickle as pickle
from model_graph import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from train_graph import layers, units, dropout, lamda, alpha, variant, cutoff, threshold


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
    dis_mat = np.load('../data/test_matrix/dismat/' + ID + '_dismat.npy')

    adj_mat = dis_mat
    for i in range(len(dis_mat)):
        for j in range(len(dis_mat)):
            if i == j: adj_mat[i][j] = 0
            elif dis_mat[i][j] <= cutoff: adj_mat[i][j] = 1
            else: adj_mat[i][j] = 0
    left_up = normalize_adjacency_matrix(adj_mat[:pro_length, :pro_length])
    right_down = normalize_adjacency_matrix(adj_mat[pro_length:, pro_length:])
    up = np.concatenate((left_up, np.zeros((left_up.shape[0], right_down.shape[0]))), axis=1)
    down = np.concatenate((np.zeros((right_down.shape[0], left_up.shape[0])), right_down), axis=1)
    norm_adj_mat = np.concatenate((up, down))

    np.save('../data/test_matrix/adjmat/'+ ID +'_adjmat.npy', norm_adj_mat)
    return norm_adj_mat


def test(pair_num_pro, pair_num_pep):
    model.eval()
    Pro_AUROC, Pro_MCC, Pep_AUROC, Pep_MCC = 0, 0, 0, 0
    with torch.no_grad():
        for batch in range(0, len(fea)):
            node_fea = fea[batch]
            node_fea = torch.FloatTensor(node_fea).to(device)
            pro_length = pro_len_list[batch]
            adj_mat = get_adjacency_matrix(dic[batch], cutoff, pro_length)
            adj_mat = torch.FloatTensor(adj_mat).to(device)
            label = lab[batch]
            label = torch.FloatTensor(label).to(device)

            node_fea_pro, node_fea_pep = node_fea[:pro_length], node_fea[pro_length:]
            adj_mat_pro, adj_mat_pep = adj_mat[:pro_length, :pro_length], adj_mat[pro_length:, pro_length:]
            label = label.view(-1, 1)
            label_pro, label_pep = label[:pro_length], label[pro_length:]

            output_pro = model(node_fea_pro, adj_mat_pro)
            output_pep = model(node_fea_pep, adj_mat_pep)
            output_pro, output_pep = torch.sigmoid(output_pro), torch.sigmoid(output_pep)
            output_pro, output_pep = output_pro.cpu().detach().numpy(), output_pep.cpu().detach().numpy()
            label_pro, label_pep = label_pro.cpu().detach().numpy(), label_pep.cpu().detach().numpy()

            TP,TN,FP,FN = 0.,0.,0.,0.
            for pro_index in range(len(label_pro)):
                pro_pred, pro_label = (output_pro[pro_index]>threshold).astype(int), label_pro[pro_index].astype(int)
                TP += (pro_pred&pro_label)
                TN += ((pro_pred==0)&(pro_label==0))
                FP += (pro_pred&(pro_label==0))
                FN += ((pro_pred==0)&pro_label)
            if (TP+FP)==0 or (TN+FN)==0:
                mcc, au_roc = 0, 0
                pair_num_pro -= 1
            else:
                mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                fpr, tpr, _ = roc_curve(label_pro, output_pro)
                au_roc = auc(fpr, tpr)
            Pro_MCC = Pro_MCC + mcc
            Pro_AUROC = Pro_AUROC + au_roc

            TP,TN,FP,FN = 0.,0.,0.,0.
            for pep_index in range(len(label_pep)):
                pep_pred, pep_label = (output_pep[pep_index]>threshold).astype(int), label_pep[pep_index].astype(int)
                TP += (pep_pred&pep_label)
                TN += ((pep_pred==0)&(pep_label==0))
                FP += (pep_pred&(pep_label==0))
                FN += ((pep_pred==0)&pep_label)
            if (TP+FP)==0 or (TN+FN)==0 or (TP+FN)==0 or (TN+FP)==0:
                mcc, au_roc = 0, 0
                pair_num_pep -= 1
            else:
                mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                fpr, tpr, _ = roc_curve(label_pep, output_pep)
                au_roc = auc(fpr, tpr)
            Pep_MCC = Pep_MCC + mcc
            Pep_AUROC = Pep_AUROC + au_roc

    return Pro_AUROC, Pro_MCC, Pep_AUROC, Pep_MCC, pair_num_pro, pair_num_pep



if __name__ == '__main__':
    SetName = 'testset'
    print('Load '+SetName+'...')
    with open('../data/settest_fea.pkl','rb') as f:
        fea = pickle.load(f)
    with open('../data/settest_lab.pkl','rb') as f:
        lab = pickle.load(f)
    dic = np.load('../data/settest_dict.npy')
    pro_len_list = np.load('../data/settest_prolen.npy')
    pep_len_list = np.load('../data/settest_peplen.npy')
    print('Load done.')

    # disrupting data
    random.seed(823)
    np.random.seed(823)
    torch.manual_seed(823)
    torch.cuda.manual_seed(823)
    pair_num = fea.shape[0]
    random_index = torch.randperm(pair_num)
    fea, lab, dic = fea[random_index], lab[random_index], dic[random_index]
    pro_len_list, pep_len_list = pro_len_list[random_index], pep_len_list[random_index]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = deepGCN(nlayers=8,
                    nfeat=fea[0][0].shape[0],
                    nhidden=512,
                    nclass=1,
                    dropout=dropout,
                    lamda=lamda,
                    alpha=alpha,
                    variant=variant).to(device)


    pair_num_pro, pair_num_pep = pair_num, pair_num
    model_file = '../model/modelgraph0.8_0.1_300_512_8.dat'
    model.load_state_dict(torch.load(model_file),False)
    Pro_AUROC, Pro_MCC, Pep_AUROC, Pep_MCC, pair_num_pro, pair_num_pep = test(pair_num_pro, pair_num_pep)

    Avg_Pro_AUROC = Pro_AUROC/pair_num_pro
    Avg_Pro_MCC = Pro_MCC/pair_num_pro
    Avg_Pep_AUROC = Pep_AUROC/pair_num_pep
    Avg_Pep_MCC = Pep_MCC/pair_num_pep

    print('Protein:'+'\tAUROC='+'%.4f'%Avg_Pro_AUROC+'\tMCC='+'%.4f'%Avg_Pro_MCC)
    print('Peptide:'+'\tAUROC='+'%.4f'%Avg_Pep_AUROC+'\tMCC='+'%.4f'%Avg_Pep_MCC)


