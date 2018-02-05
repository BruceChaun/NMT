import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Dense(nn.Module):

    def __init__(self, layers):
        nn.Module.__init__(self)

        self.fc = nn.ModuleList(
                [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
                )


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
        self.out = Dense([hid_dim * 3, vocab_size])


    def forward(self, x, h, encoder_out):
        x = self.embedding(x)
        batch_size, encoder_len, dim = encoder_out.size()

        betas = Variable(torch.zeros(batch_size, encoder_len))
        if x.is_cuda:
            betas = betas.cuda()

        last_hidden = h[-1]
        for i in range(encoder_len):
            betas[:,i] = F.softmax(self.attn(
                torch.cat((last_hidden, encoder_out[:,i,:]), dim=1)
                ))
        attn_weights = F.softmax(betas).unsqueeze(dim=1)
        context = attn_weights.bmm(encoder_out)

        rnn_input = torch.cat((x, context), dim=2)
        self.gru.flatten_parameters()
        output, h = self.gru(rnn_input, h)
        output = F.log_softmax(self.out(
            torch.cat((output.squeeze(dim=1), context.squeeze(dim=1)), dim=1)
            ))

        return output, h, attn_weights

