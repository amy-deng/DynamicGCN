# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_sparse import spmm
from utils import *



class SpGraphConvLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SpGraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
#        support = torch.spmm(feature, self.weight) # sparse 
#        output = torch.spmm(adj, support)
        support = torch.mm(feature, self.weight) # sparse
        output = spmm(adj._indices(),adj._values(), adj.size(0), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MaskLinear(Module):
    def __init__(self, in_features, out_features=1, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.Tensor(in_features)) 
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, idx): # idx is a list
        mask = torch.zeros(self.in_features).cuda()
        mask[idx] = x.squeeze()
        output = torch.matmul(self.weight, mask)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' => ' \
               + str(self.out_features) + ')'



class TemporalEncoding(Module):
    def __init__(self, in_features, out_features, bias=True): 
        super(TemporalEncoding, self).__init__()
        out_features = int(in_features / 2) # not useful now
        out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, out_o))  
        self.weight_c = Parameter(torch.Tensor(in_features, out_c))
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_c.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features)) 
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_o, h_c):  
        trans_ho = torch.mm(h_o, self.weight_o)  
        trans_hc = torch.mm(h_c, self.weight_c)  
        output =torch.tanh( (torch.cat((trans_ho, trans_hc), dim=1))) # dim=1

        if self.bias is not None:
            return output + self.bias
        else:
            return output
