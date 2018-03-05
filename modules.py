import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

import numpy as np
import math


def position_embedding(num_pos, d):
    '''
    sinusoidal positional embedding
    '''
    pos_emb = np.array([
            [pos / np.power(10000, 2.*i/d) for i in range(d)]
            for pos in range(num_pos)])

    pos_emb[:,0::2] = np.sin(pos_emb[:,0::2])
    pos_emb[:,0::2] = np.cos(pos_emb[:,0::2])

    pos_emb = Variable(torch.FloatTensor(pos_emb))
    if torch.cuda.is_available():
        pos_emb = pos_emb.cuda()
    return pos_emb


def get_padding_mask(Q, K):
    batch_size, q_len = Q.size()
    k_len = K.size(1)
    mask = K.data.eq(0).unsqueeze(1)
    mask = mask.expand(batch_size, q_len, k_len)
    return mask


class ScaledDotProductAttn(nn.Module):

    def __init__(self, d):
        nn.Module.__init__(self)

        self.d = d
        self.scale = math.sqrt(1. / d)


    def forward(self, Q, K, V, mask=None):
        '''
        Q, K, V have the dimension of (batch_size, seq_len, dim)
        mask: (batch_size, seq_len, seq_len), mask out illegal positions
        '''
        attn = torch.bmm(Q, K.transpose(1,2).contiguous()) * self.scale
        if mask is not None:
            assert attn.size() == mask.size()
            attn.data.masked_fill_(mask, -float('inf'))

        attn = F.softmax(attn, dim=2)
        attn = attn.bmm(V)
        return attn


class MultiHeadAttn(nn.Module):

    def __init__(self, 
            num_head, 
            d_model, 
            d_k, 
            d_v, 
            dropout):
        '''
        num_head: int, number of heads
        d_model: int
        d_k: int, dimension of query and key
        d_v: int, dimension of value
        dropout: float, dropout rate
        '''
        nn.Module.__init__(self)
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        self.Wq = nn.ModuleList(
                [nn.Linear(d_model, d_k) for n in range(num_head)])
        self.Wk = nn.ModuleList(
                [nn.Linear(d_model, d_k) for n in range(num_head)])
        self.Wv = nn.ModuleList(
                [nn.Linear(d_model, d_v) for n in range(num_head)])
        self.Wo = nn.Linear(d_v * num_head, d_model)

        self.attn = ScaledDotProductAttn(d_model)
        self.bn = nn.BatchNorm1d(d_model)

        self.__init_weights()


    def __init_weights(self):
        for w in self.Wq:
            init.xavier_normal(w.weight)

        for w in self.Wk:
           init.xavier_normal(w.weight)
           
        for w in self.Wv:
            init.xavier_normal(w.weight)

        init.xavier_normal(self.Wo.weight)


    def forward(self, Q, K, V, mask=None):
        '''
        Q, K, V: (batch_size, seq_len, dim)
        mask: (batch_size, seq_len, seq_len), mask out illegal positions
        
        NOTE: return a tensor with shape (batch_size, dim, seq_len)
        '''
        residual = Q

        out = []
        for i in range(self.num_head):
            out.append(self.attn(
                self.Wq[i](Q), self.Wk[i](K), self.Wv[i](V), mask)
                )
        out = torch.cat(out, dim=2)

        out = self.Wo(out)
        out = F.dropout(out, self.dropout, self.training)
        out += residual
        out = self.bn(out.transpose(1,2).contiguous())

        return out


class PositionWiseFeedFoward(nn.Module):

    def __init__(self, input_dim, hid_dim, dropout):
        '''
        input_dim: int, dimension of input tensor
        hid_dim: int, dimension of hidden state
        dropout: float, dropout rate
        '''
        nn.Module.__init__(self)

        self.dropout = dropout
        self.fc1 = nn.Conv1d(input_dim, hid_dim, 1)
        self.fc2 = nn.Conv1d(hid_dim, input_dim, 1)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        '''
        x: (batch_size, dim, seq_len)

        NOTE: return a tensor with shape (batch_size, seq_len, dim)
        '''
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.bn(x + residual).transpose(1,2).contiguous()
        return x

