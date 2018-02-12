import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

import math
import hashlib


class N_Gram_Embedding(nn.Module):

    def __init__(self, 
            vocab, 
            vocab_sizes, 
            emb_size):
        '''
        vocab: dict, (word, index) pairs
        vocab_sizes: list, size of each n-gram
        emb_size: int
        '''
        nn.Module.__init__(self)

        self.ivocab = {i:w for w,i in vocab.items()}
        self.vocab_sizes = vocab_sizes
        self.emb_size = emb_size
        self.n = len(vocab_sizes)
        # special token embedding and n_gram embeddings
        self.embeddings = nn.ModuleList(
                [nn.Embedding(4, emb_size*self.n, padding_idx=0)] + 
                [nn.Embedding(size+1, emb_size, padding_idx=0) for size in vocab_sizes]
                )

        # initialize weights
        for i in range(len(self.embeddings)):
            self.embeddings[i].weight.data.normal_(0, 0.1)


    def _hash(self, s):
        V = self.vocab_sizes[len(s)-1]
        v = int(hashlib.sha256(s.encode()).hexdigest(), 16)
        return v % V + 1


    def _word_embedding(self, word):
        cuda = torch.cuda.is_available()
        n_grams = []
        for i in range(1, 1 + self.n):
            i_grams = [self._hash(word[j:j+i]) for j in range(len(word)-i+1)]
            if len(i_grams) == 0:
                i_grams = [0]

            i_grams = Variable(torch.LongTensor(i_grams))
            if cuda:
                i_grams = i_grams.cuda()
            emb = self.embeddings[i](i_grams).mean(0)
            n_grams.append(emb)

        return torch.cat(n_grams)


    def forward(self, x):
        '''
        x: Tensor, (batch_size, seq_len)
        cuda: boolean
        '''
        batch_size, seq_len = x.size()
        embeddings = Variable(torch.zeros(batch_size, seq_len, self.emb_size*self.n))
        if x.is_cuda:
            embeddings = embeddings.cuda()

        for b in range(batch_size):
            for i in range(seq_len):
                wi = x[b,i].data.cpu().numpy()[0]
                if wi < 4:
                    embeddings[b,i,:] = self.embeddings[0](
                            x[b,i].view(1,1)).squeeze()
                else:
                    word = self.ivocab[wi]
                    embeddings[b,i,:] = self._word_embedding(word)

        return embeddings


class RNNEncoder(nn.Module):

    def __init__(self, 
            emb_size, 
            hid_dim, 
            vocab_sizes, 
            vocab, 
            n_layers, 
            dropout):
        '''
        emb_size: int, embedding size of n-grams
        hid_dim: int, size of rnn hidden layer
        vocab_sizes: list, size of n-grams
        vocab: dict, (word, index) pairs
        n_layers: int, number of rnn layers
        dropout: float, rnn dropout
        '''
        nn.Module.__init__(self)

        self.embedding = N_Gram_Embedding(vocab, vocab_sizes, emb_size)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(emb_size*len(vocab_sizes), hid_dim, n_layers, 
                batch_first=True, dropout=dropout, bidirectional=True)


    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers*2, batch_size, self.hid_dim))


    def forward(self, x, h, seq_len):
        x = self.embedding(x)
        packed_seq = pack_padded_sequence(x, seq_len, batch_first=True)
        self.gru.flatten_parameters()
        output, h = self.gru(packed_seq, h)
        output, out_len = pad_packed_sequence(output, True)
        return output, h

##
# Credit
# https://github.com/facebookresearch/fairseq-py/blob/master/fairseq/modules/grad_multiply.py
##
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class CNNEncoder(nn.Module):

    def __init__(self, 
            emb_size, 
            vocab_sizes, 
            vocab, 
            kernels, 
            decoder_attn_layers, 
            dropout):
        '''
        emb_size: int, embedding size of n-grams
        vocab_sizes: list of int, vocabulary size of n-grams
        vocab: dict, (word, index) pairs
        kernels: list of int, each entry indicates one layer of convolution
        decoder_attn_layers: int, number of attention layers in decoder
        dropout: float
        '''
        nn.Module.__init__(self)

        self.kernels = kernels
        self.dropout = dropout
        self.attn_layers = decoder_attn_layers

        self.embedding = N_Gram_Embedding(vocab, vocab_sizes, emb_size)
        word_dim = emb_size * len(vocab_sizes)

        self.convs = []
        for k in kernels:
            conv = nn.Conv1d(word_dim, word_dim * 2, k, padding=k//2)
            std = math.sqrt(4. * (1 - dropout) / (k * word_dim))
            conv.weight.data.normal_(0, std)
            conv.bias.data.zero_()
            self.convs.append(nn.utils.weight_norm(conv, dim=2))

        self.convs = nn.ModuleList(self.convs)


    def forward(self, x):
        e = self.embedding(x).transpose(1,2)
        x = e
        n_layers = len(self.kernels)

        for i in range(n_layers):
            residual = x
            x = F.dropout(x, self.dropout, self.training)

            x = self.convs[i](x)
            x = F.glu(x, 1)
            x = GradMultiply.apply(x, 1.0 / (2.0 * self.attn_layers))
            x = (x + residual) * math.sqrt(0.5)

        return x, e

