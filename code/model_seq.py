import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        # theta = math.log(lamda/l+1)
        theta = min(1, math.log(lamda/l+1))
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual: # speed up convergence of the training process
            output = output+input
        return output


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, residual=True, variant=variant))
        self.trans_encoderlayer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=64*4, dropout=dropout)
        self.trans_encoder = nn.TransformerEncoder(self.trans_encoderlayer, num_layers=2)
        self.fcs_gra = nn.ModuleList()
        self.fcs_gra.append(nn.Linear(nfeat, nhidden))
        self.fcs_gra.append(nn.Linear(nhidden, nclass))
        self.fcs_tra = nn.ModuleList()
        self.fcs_tra.append(nn.Linear(nfeat-12, 64))
        self.fcs_tra.append(nn.Linear(64, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha

    def forward(self, pro_gra, pro_adj, pep_tra):
        _layers = []
        pro_gra = F.dropout(pro_gra, self.dropout, training=self.training)
        pro_gra = self.act_fn(self.fcs_gra[0](pro_gra))
        _layers.append(pro_gra)
        for i, con in enumerate(self.convs):
            pro_gra = F.dropout(pro_gra, self.dropout, training=self.training)
            pro_gra = self.act_fn(con(pro_gra, pro_adj, _layers[0], self.lamda, self.alpha, i+1))
        pro_gra = F.dropout(pro_gra, self.dropout, training=self.training)
        pro_gra = self.fcs_gra[-1](pro_gra)
        
        # remove dssp
        pep_tra_pssm_hmm, pep_tra_property_pretrain = pep_tra[:,:50], pep_tra[:,62:]
        pep_tra = torch.cat([pep_tra_pssm_hmm, pep_tra_property_pretrain], dim=1)
        pep_tra = F.dropout(pep_tra, self.dropout, training=self.training)
        pep_tra = self.act_fn(self.fcs_tra[0](pep_tra))
        pep_tra = pep_tra.unsqueeze(0).permute(1,0,2)
        pep_tra = self.trans_encoder(pep_tra)
        pep_tra = pep_tra.permute(1,0,2).squeeze(0)
        pep_tra = F.dropout(pep_tra, self.dropout, training=self.training)
        pep_tra = self.fcs_tra[-1](pep_tra)

        layer_inner = torch.cat((pro_gra, pep_tra))
        return layer_inner






