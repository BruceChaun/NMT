import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math

from modules import *


class Dense(nn.Module):

    def __init__(self, layers):
        nn.Module.__init__(self)

        self.fc = nn.ModuleList(
                [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
                )

        #for i in range(len(layers)-1):
        #    self.fc[i].weight.data.normal_(0, math.sqrt(1. / layers[i]))


    def forward(self, x):
        for fc in self.fc:
            x = fc(x)
        return x


class RNNDecoder(nn.Module):

    def __init__(self, 
            emb_size, 
            hid_dim, 
            vocab_size, 
            n_layers, 
            dropout):
        '''
        emb_size: int
        hid_dim: int, size of rnn hidden layer
        vocab_size: int
        n_layers: int, number of rnn layers
        dropout: float, rnn dropout
        '''
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)

        self.gru = nn.GRU(hid_dim * 2 + emb_size, hid_dim, n_layers, 
                batch_first=True, dropout=dropout)

        self.attn = nn.Linear(hid_dim * 3, 1)
        self.out = nn.Linear(hid_dim * 3, vocab_size)

        # init weights
        self.embedding.weight.data.normal_(0, 0.1)
        for p in self.gru.parameters():
            p.data.normal_(0, 0.01)
        self.attn.weight.data.normal_(0, 0.01)
        self.out.weight.data.normal_(0, 0.01)


    def forward(self, x, h, encoder_out):
        '''
        x: LongTensor, current input, (batch_size, 1)
        h: FloatTensor, previous hidden state of decoder
        encoder_out: FloatTensor, output representation of encoder, 
           (batch_size, seq_len, feature_dim)
        '''
        x = self.embedding(x)
        batch_size, encoder_len, dim = encoder_out.size()

        betas = Variable(torch.zeros(batch_size, encoder_len))
        if x.is_cuda:
            betas = betas.cuda()

        last_hidden = h[-1]
        for i in range(encoder_len):
            betas[:,i] = self.attn(
                torch.cat((last_hidden, encoder_out[:,i,:]), dim=1)
                )
        attn_weights = F.softmax(betas).unsqueeze(dim=1)
        context = attn_weights.bmm(encoder_out)

        rnn_input = torch.cat((x, context), dim=2)
        self.gru.flatten_parameters()
        output, h = self.gru(rnn_input, h)
        output = F.log_softmax(self.out(
            torch.cat((output.squeeze(dim=1), context.squeeze(dim=1)), dim=1)
            ))

        return output, h, attn_weights


class CNNDecoder(nn.Module):

    def __init__(self, 
            emb_size, 
            vocab_size, 
            kernels, 
            dropout):
        '''
        emb_size: int
        vocab_size: int
        kernels: list of int
        dropout: float
        '''
        nn.Module.__init__(self)

        self.emb_size = emb_size
        self.kernels = kernels
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(emb_size, vocab_size)

        self.convs = []
        for k in kernels:
            conv = nn.Conv1d(emb_size, emb_size * 2, k, padding=k-1)
            std = math.sqrt(4. * (1 - dropout) / (k * emb_size))
            conv.weight.data.normal_(0, std)
            conv.bias.data.zero_()
            self.convs.append(nn.utils.weight_norm(conv, dim=2))

        self.convs = nn.ModuleList(self.convs)


    def forward(self, x, encoder_out):
        '''
        x: previously generated elements by decoder
        encoder_out: FloatTensor, final layer output of encoder
        '''
        target = self.embedding(x).transpose(1,2)
        x = target
        n_layers = len(self.kernels)

        for l in range(n_layers):
            k = self.kernels[l]
            residual = x

            # Eq. 1
            x = F.dropout(x, self.dropout, self.training)
            x = self.convs[l](x)[:,:,:1-k]
            x = F.glu(x, 1)
            residual_attn = x
            x = (x + target) * math.sqrt(0.5)

            # Eq. 2, attention
            x = x.transpose(1, 2)
            attn = x.bmm(encoder_out)
            sz = attn.size()
            attn = F.softmax(attn.view(sz[0] * sz[1], sz[2]), dim=1)
            attn = attn.view(sz).transpose(1,2)

            # Eq. 3, conditional input
            x = torch.bmm(encoder_out, attn)
            x = x * math.sqrt(encoder_out.size(2))

            # residual
            x = (x + residual_attn) * math.sqrt(0.5)
            x = (x + residual) * math.sqrt(0.5)

        del target, residual, residual_attn, attn

        out = F.log_softmax(self.out(x[:,:,-1]), dim=1)
        return out


class AttnDecoder(nn.Module):

    def __init__(self, 
            emb_size, 
            hid_dim, 
            d_k, 
            d_v, 
            num_head, 
            vocab_size, 
            num_attn_layers, 
            dropout, 
            tar_len=200):
        '''
        emb_size: int, word embedding size, same as d_model in the paper
        hid_dim: int, hidden dimension in position wise feed forward layer
        d_k: int, dimension of query and key
        d_v: int, dimension of value
        num_head: int, number of heads in multi-head attention layer
        vocab_size: int, vocabulary size of target language sentence
        num_attn_layers: int, number of attention layers
        dropout: float, dropout rate in the network
        tar_len: int, max length of translated sentence
        '''
        nn.Module.__init__(self)
        self.num_attn_layers = num_attn_layers
        self.dropout = dropout
        self.pos_emb = position_embedding(tar_len, emb_size)

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(emb_size, vocab_size, False)
        self.out.weight = self.embedding.weight

        self.self_attn = nn.ModuleList([
            MultiHeadAttn(num_head, emb_size, d_k, d_v, dropout)
            for i in range(num_attn_layers)])
        self.encoder_attn = nn.ModuleList([
            MultiHeadAttn(num_head, emb_size, d_k, d_v, dropout)
            for i in range(num_attn_layers)])
        self.ff = nn.ModuleList([
             PositionWiseFeedFoward(emb_size, hid_dim, dropout)
             for i in range(num_attn_layers)])


    def forward(self, src_seq, decoder_input, encoder_out):
        '''
        src_seq: (batch_size, seq_len)
        decoder_input: (batch_size, seq_len)
        encoder_out: (batch_size, seq_len, feature_dim)
        '''
        batch_size, dec_len = decoder_input.size()
        dec_self_attn_mask = get_padding_mask(decoder_input, decoder_input)
        enc_dec_attn_mask = get_padding_mask(decoder_input, src_seq)

        x = self.embedding(decoder_input) + self.pos_emb[:dec_len]
        for i in range(self.num_attn_layers):
            x = self.self_attn[i](x, x, x, dec_self_attn_mask)
            x = self.encoder_attn[i](
                    x.transpose(1,2), encoder_out, encoder_out, enc_dec_attn_mask)
            x = self.ff[i](x)

        x = self.out(x[:,-1,:])
        return x
