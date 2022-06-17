import numpy as np
import scipy
from scipy import io as sio
from scipy.sparse import csr_matrix
import random
from gym.spaces import Discrete
from gym import spaces
from copy import deepcopy
import sys
import gc

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import spspmm

import torch_geometric as tg
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_geometric.data import Dataset, DataLoader

import dgl
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)


def array_interset(array2a, array2b):
    array2a = array2a.transpose()
    array2b = array2b.transpose()
    a = set((tuple(i) for i in array2a))
    b = set((tuple(i) for i in array2b))
    
    return np.array(list(a.intersection(b))).transpose()


def array_diffset(array2a, array2b):
    array2a = array2a.transpose()
    array2b = array2b.transpose()
    a = set((tuple(i) for i in array2a))
    b = set((tuple(i) for i in array2b))
    
    return np.array(list(a.difference(b))).transpose()


class HGNN(nn.Module):
    def __init__(self, pretrain_gnn, walk_length, num_relations, feat_dim, hid_dim, out_dim, gnn_type, gnn_layers,
                 agg_type, rnn_type, dropout, act_type, device):
        super(HGNN, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.walk_length = walk_length
        self.num_relations = num_relations
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers
        self.agg_type = agg_type
        self.rnn_type = rnn_type
        self.num_head = 8
        self.dropout = dropout
        self.act_type = act_type # 0-relu 1-elu
        self.device = device
        self.pretrain_gnn = pretrain_gnn

        # self.mlp = nn.Sequential(nn.Linear(self.feat_dim, 3*self.hid_dim), nn.ReLU(),
        #                         nn.Linear(3*self.hid_dim, 2*self.hid_dim), nn.ReLU(), 
        #                         nn.Linear(2*self.hid_dim, self.hid_dim))
        self.mlp = nn.Linear(self.feat_dim, hid_dim)
        
        if self.act_type == 'relu':
            self.act_f = F.relu
        elif self.act_type == 'elu':
            self.act_f = F.elu
        elif self.act_type == 'NONE':
            self.act_f = lambda x: x
        else:
            print('Wrong activation')  

        if self.gnn_type != 'NONE':
            self.gnns = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            for _ in range(self.gnn_layers):
                r_gnn = nn.ModuleList()
                if self.gnn_type == 'GCN':
                    for _ in range(self.num_relations):
                        r_gnn.append(GCNConv(self.hid_dim, self.hid_dim))
                elif self.gnn_type == 'GAT':
                    for _ in range(self.num_relations):
                        r_gnn.append(GATConv(
                            self.hid_dim, self.hid_dim // self.num_head,
                            heads=self.num_head, dropout=self.dropout
                        ))
                self.gnns.append(r_gnn)
                self.batch_norms.append(BatchNorm(self.hid_dim))

        if self.rnn_type != 'NONE':
            self.attn = nn.Parameter(torch.Tensor(size=(1, self.num_head, self.hid_dim)))
            nn.init.xavier_normal_(self.attn.data, gain=1.414)
            self.leaky_relu = nn.LeakyReLU(0.01)
            self.softmax = edge_softmax
            self.attn_drop = nn.Dropout(self.dropout)
            self.lin_rnn_emb = nn.Linear(self.hid_dim * self.num_head, self.hid_dim, bias=True)
            nn.init.xavier_normal_(self.lin_rnn_emb.weight, gain=1.414)

            if self.rnn_type == 'gru':
                self.rnn = nn.GRU(self.hid_dim, self.num_head * self.hid_dim)
            elif self.rnn_type == 'lstm':
                self.rnn = nn.LSTM(self.hid_dim, self.num_head * self.hid_dim)
            elif self.rnn_type == 'bi-gru':
                self.rnn = nn.GRU(self.hid_dim, self.num_head * self.hid_dim // 2, bidirectional=True)
            elif self.rnn_type == 'bi-lstm':
                self.rnn = nn.LSTM(self.hid_dim, self.num_head * self.hid_dim // 2, bidirectional=True)
        
        # last linear layer
        self.last_layer = nn.Linear(self.hid_dim, self.out_dim, bias=True)
        nn.init.xavier_normal_(self.last_layer.weight, gain=1.414)
    
    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def rnn_channel(self, embedding, rnn_paths):
        flat_array = np.array([item[::-1] for sublist in rnn_paths for item in sublist])
        # b = flat_array[flat_array.min(axis=1)>=0, :]
        # print('rnn:', b.shape[0]*(b.shape[1]-1))
        for idx, m in enumerate(flat_array):
            if -1 in m:
                m[m == -1] = m[(m != -1).argmax(axis=0)]
                flat_array[idx] = m
        # rnn_data = flat_array[~np.any(flat_array == -1, axis=1)]
        # related_nodes = np.unique(rnn_data[:, -1])
        edge_metapath_indices = torch.LongTensor(flat_array).to(self.device)
        g = dgl.graph((edge_metapath_indices[:, 0], edge_metapath_indices[:, -1]))
        
        edata = F.embedding(edge_metapath_indices, embedding)
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.hid_dim, self.num_head).permute(0, 2, 1).reshape(
                -1, self.num_head * self.hid_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.hid_dim, self.num_head).permute(0, 2, 1).reshape(
                -1, self.num_head * self.hid_dim).unsqueeze(dim=0)
        
        eft = hidden.permute(1, 0, 2).view(-1, self.num_head, self.hid_dim)  # E x num_heads x out_dim
        a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # E x num_heads x out_dim
        ret = ret.view(-1, self.num_head * self.hid_dim)
        
        embedding = self.lin_rnn_emb(ret)
        return embedding

    def agg_paths2r_paths(self, data, agg_paths, quiet):
        agg_paths = [item for item in agg_paths if len(item) > 0]
        # agg all paths one time
        agg_edge = torch.LongTensor(np.unique(torch.cat(agg_paths, axis=-1).cpu().numpy(), axis=1)).to(self.device)
        # for agg_edge in agg_paths:
        if data.data_name == 'IMDB':
            # 0-MD, 1-DM, 2-MA, 3-AM
            r_agg_paths = [[] for _ in range(self.num_relations)]
            r_agg_paths[0] = torch.LongTensor(array_interset(
                np.array(data.edge_index_md.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[1] = torch.LongTensor(array_interset(
                np.array(data.edge_index_dm.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[2] = torch.LongTensor(array_interset(
                np.array(data.edge_index_ma.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[3] = torch.LongTensor(array_interset(
                np.array(data.edge_index_am.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            if not quiet:
                print('-' * 50)
                for rtype, item in zip(['MD', 'DM', 'MA', 'AM'], r_agg_paths):
                    print('All step {}'.format(len(agg_paths)), rtype, item.shape)
        elif data.data_name == 'DBLP':
            # 0-AP, 1-PA, 2-PT, 3-TO, 4-PC, 5-CP
            r_agg_paths = [[] for _ in range(self.num_relations)]
            r_agg_paths[0] = torch.LongTensor(array_interset(
                np.array(data.edge_index_ap.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[1] = torch.LongTensor(array_interset(
                np.array(data.edge_index_pa.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[2] = torch.LongTensor(array_interset(
                np.array(data.edge_index_pt.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[3] = torch.LongTensor(array_interset(
                np.array(data.edge_index_tp.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[4] = torch.LongTensor(array_interset(
                np.array(data.edge_index_pc.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            r_agg_paths[5] = torch.LongTensor(array_interset(
                np.array(data.edge_index_cp.tolist()), np.array(agg_edge.tolist())
            )).to(self.device)
            if not quiet:
                print('-' * 50)
                for rtype, item in zip(['AP', 'PA', 'PT', 'TP', 'PC', 'CP'], r_agg_paths):
                    print('All step {}'.format(len(agg_paths)), rtype, item.shape)
        return r_agg_paths

    def gnn_channel(self, data, x, agg_paths, quiet):
        r_agg_paths = self.agg_paths2r_paths(data, agg_paths, quiet)
        residual = x.clone()

        # for r_gnn, batch_norm in zip(self.gnns[:-1], self.batch_norms[:-1]):
        for r_gnn, batch_norm in zip(self.gnns, self.batch_norms):
            h_res = []
            if data.data_name == 'IMDB':
                if r_agg_paths[0].shape[0] > 0:
                    h_res.append(r_gnn[0](x, r_agg_paths[0]))
                if r_agg_paths[1].shape[0] > 0:
                    h_res.append(r_gnn[1](x, r_agg_paths[1]))
                if r_agg_paths[2].shape[0] > 0:
                    h_res.append(r_gnn[2](x, r_agg_paths[2]))
                if r_agg_paths[3].shape[0] > 0:
                    h_res.append(r_gnn[3](x, r_agg_paths[3]))
            elif data.data_name == 'DBLP':
                if r_agg_paths[0].shape[0] > 0:
                    h_res.append(r_gnn[0](x, r_agg_paths[0]))
                if r_agg_paths[1].shape[0] > 0:
                    h_res.append(r_gnn[1](x, r_agg_paths[1]))
                if r_agg_paths[2].shape[0] > 0:
                    h_res.append(r_gnn[2](x, r_agg_paths[2]))
                if r_agg_paths[3].shape[0] > 0:
                    h_res.append(r_gnn[3](x, r_agg_paths[3]))
                if r_agg_paths[4].shape[0] > 0:
                    h_res.append(r_gnn[4](x, r_agg_paths[4]))
                if r_agg_paths[5].shape[0] > 0:
                    h_res.append(r_gnn[5](x, r_agg_paths[5]))
            # stack all relations by mean/sum
            # x = self.act_f(batch_norm(torch.stack(h_res, dim=-1).sum(dim=-1)))
            x = self.act_f(batch_norm(torch.stack(h_res, dim=-1).mean(dim=-1)))
            x = x + residual  # residual
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def self_channel(self, data):
        h = self.mlp(data.x)
        return h

    def forward(self, data, agg_paths, rnn_paths, quiet=False):
        if self.pretrain_gnn:
            embedding = data.pre_x
        else:
            embedding = self.self_channel(data)

        if self.gnn_type != 'NONE':
            if len(agg_paths) > 0:
                if agg_paths[0].shape[1]>0:
                    embedding = self.gnn_channel(data, embedding, agg_paths, quiet)
        if self.rnn_type != 'NONE':
            if len(agg_paths) > 0:
                if agg_paths[0].shape[1] > 0:
                    rnn_embedding = self.rnn_channel(embedding, rnn_paths)
                    if rnn_embedding.shape[0] < embedding.shape[0]:
                        to_fill = embedding.shape[0]-rnn_embedding.shape[0]
                        rnn_embedding = torch.cat([rnn_embedding, embedding[-(to_fill):]], axis=0)
                    embedding = rnn_embedding
        
        logic = self.last_layer(embedding)
        return F.log_softmax(logic, dim=1), F.softmax(logic, dim=1), embedding
