# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
from layers import *


class GCN(nn.Module):
    def __init__(self, pretrained_emb, n_output, dropout, instance_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)

        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = False
        self.n_feature = pretrained_emb.size(1)
        self.n_input = pretrained_emb.size(0)

        self.conv1 = SpGraphConvLayer(self.n_feature, self.n_feature)
        self.conv1_bn = nn.BatchNorm1d(self.n_feature)
        self.conv2 = SpGraphConvLayer(self.n_feature, n_output)
        self.conv2_bn = nn.BatchNorm1d(n_output)
        self.mask = MaskLinear(self.n_input, n_output)
        self.save_x = None

    def forward(self, adj, vertices): 
        emb = self.embedding(vertices)
        if self.instance_norm:
            emb = self.norm(emb)
        x = emb
        x = F.relu(self.conv1_bn(self.conv1(x, adj)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2_bn(self.conv2(x, adj)))
        x = self.mask(x, vertices)
        x = torch.sigmoid(x)
        return x, self.save_x


class DynamicGCN(nn.Module):
    def __init__(self, pretrained_emb, n_output, n_hidden=7, dropout=0.2, instance_norm=False):
        super(DynamicGCN, self).__init__()

        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)

        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = False
        self.n_feature = pretrained_emb.size(1)
        self.n_input = pretrained_emb.size(0)


        self.layer_stack = nn.ModuleList() # TODO class initiate
        self.bn_stack = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden-1):
            self.layer_stack.append(SpGraphConvLayer(self.n_feature, self.n_feature))
            self.bn_stack.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding(self.n_feature,self.n_feature))
            
        self.layer_stack.append(SpGraphConvLayer(self.n_feature, n_output))
        self.bn_stack.append(nn.BatchNorm1d(n_output)) 
        self.mask = MaskLinear(self.n_input, n_output)
        self.save_x = None

    def forward(self, adjs, vertices): 
        emb = self.embedding(vertices)
        if self.instance_norm:
            emb = self.norm(emb)
        x = emb

        r = []
        for i, gcn_layer in enumerate(self.layer_stack):
            last_x = x
            x = self.bn_stack[i](gcn_layer(x, adjs[i]))
            if i < len(self.layer_stack)-1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.temporal_cells[i-1](last_x,x) # temporal encoding 
            
        x = F.relu(x)
        x = self.mask(x, vertices)
        x = torch.sigmoid(x)
        return x, None
