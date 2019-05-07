# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os
import torch
import re
import string
import torch
import torch.nn.functional as F 

 
def split_data(size, train=.7, val=.15, test=.15, shuffle=True):
    idx = list(range(size))
    if shuffle:
        np.random.shuffle(idx)
    split_idx = np.split(idx, [int(train * len(idx)), int((train+val) * len(idx))])
    train_idx, val_idx, test_idx = split_idx[0], split_idx[1], split_idx[2]
    return train_idx, val_idx, test_idx

 
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize(mx):
    """Row-normalize sparse matrix  (normalize feature)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col)==0:
        print(sparse_mx.row,sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_tensor_to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
 
def check_exist(outf):
    return os.path.isfile(outf)


def load_w2v_emb(file):
    print('load_w2v_emb',file)
    with open(file, 'rb') as f: 
        emb = np.load(f)
    return emb # np.narray type


def load_dynamic_graph_data(dataset_str, emb_str, f_dim=100, train=.825, val=.175, test=.0, shuffle=False):

    names = ['x', 'y', 'idx', 'tx', 'ty', 'tidx']
    objects = []
    
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            objects.append(np.load(f))
    x, y, idx, tx, ty, tidx= tuple(objects)

    x = [sparse_mx_to_torch_sparse_tensor(normalize_adj(xx)) for xx in x[:]]
    y = [torch.from_numpy(yy).float() for yy in y[:]] 
    idx = [torch.from_numpy(_idx).long() for _idx in idx[:]]

    tx = [sparse_mx_to_torch_sparse_tensor(normalize_adj(xx)) for xx in tx[:]]
    ty = [torch.from_numpy(yy).float() for yy in ty[:]] 
    tidx = [torch.from_numpy(_idx).long() for _idx in tidx[:]]

    train_idx, val_idx, _  = split_data(len(x), train, val, test, shuffle=False)

    train_dict, val_dict, test_dict = {}, {}, {}
    names_dict = {'x':x, 'y':y, 'idx':idx}
    for name in names_dict:
        train_dict[name] = [names_dict[name][i] for i in train_idx]
        val_dict[name] = [names_dict[name][i] for i in val_idx]
    test_dict = {'x':tx, 'y':ty, 'idx':tidx}
    emb_file = os.path.join('data/', "{}.emb_{}".format(emb_str, f_dim))
    emb = load_w2v_emb(emb_file)
    emb = torch.FloatTensor(emb)
    return train_dict, val_dict, test_dict, emb


 

def load_sparse_temporal_data(dataset_str, emb_str, f_dim, train=.825, val=.175, test=.0, shuffle=False):
    names = ['x', 'y', 'idx', 'tx', 'ty', 'tidx']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            objects.append(np.load(f,encoding='latin1'))
    x, y, idx, tx, ty, tidx= tuple(objects)
    ## train 
    for xx in x:
        for i in range(len(xx)):
            xx[i] = sparse_mx_to_torch_sparse_tensor(normalize_adj(xx[i]))

    y = [torch.from_numpy(yy).float() for yy in y[:]] 
    idx = [torch.from_numpy(_idx).long() for _idx in idx[:]]

    ## test
    for xx in tx:
        for i in range(len(xx)):
            xx[i] = sparse_mx_to_torch_sparse_tensor(normalize_adj(xx[i]))

    ty = [torch.from_numpy(yy).float() for yy in ty[:]] 
    tidx = [torch.from_numpy(_idx).long() for _idx in tidx[:]]

    train_idx, val_idx, _  = split_data(len(x), train, val, test, shuffle=False)

    train_dict, val_dict, test_dict = {}, {}, {}
    names_dict = {'x':x, 'y':y, 'idx':idx}
    for name in names_dict:
        train_dict[name] = [names_dict[name][i] for i in train_idx]
        val_dict[name] = [names_dict[name][i] for i in val_idx]
    test_dict = {'x':tx, 'y':ty, 'idx':tidx}
    emb_file = os.path.join('data/', "{}.emb_{}".format(emb_str, f_dim))
    emb = load_w2v_emb(emb_file)
    emb = torch.FloatTensor(emb)
    return train_dict, val_dict, test_dict, emb

 