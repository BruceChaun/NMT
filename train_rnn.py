import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import numpy as np

import random
import sys
import os
import pickle
import copy

from config import RNNConfig
from dataset import *
from encoders import RNNEncoder
from decoders import RNNDecoder
from eval import evaluate_rnn
import utils


def train(encoder, decoder, dataloader, conf):
    encoder.train()
    decoder.train()
    enc_opt = optim.Adam(encoder.parameters(), lr=conf.lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=conf.lr)
    loss_fn = nn.NLLLoss()

    total_loss = 0
    batches = 0
    for src, src_len, ref, ref_len in dataloader:
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        loss = 0

        src = src[:,1:] # skip <SOS>
        batch_size, src_max_len = src.shape
        src = Variable(torch.LongTensor(src))
        ref = Variable(torch.LongTensor(ref))
        enc_h = encoder.init_hidden(batch_size)

        if conf.cuda:
            src = src.cuda()
            ref = ref.cuda()
            enc_h = enc_h.cuda()

        encoder_out, enc_h = encoder(src, enc_h, src_len)
        dec_h = enc_h[-2:-1] # last forward layer in encoder

        ref_max_len = ref.size(1)
        decoder_input = ref[:,:1]
        mask = torch.Tensor(utils.mask_matrix(ref_len, ref_max_len))

        teacher_forcing = random.random() < conf.teaching_ratio

        if teacher_forcing:
            for i in range(1, ref_max_len):
                decoder_out, dec_h, attn_weights = decoder(
                        decoder_input, dec_h, encoder_out)
                loss += utils.batch_loss(loss_fn, decoder_out, ref[:,i], mask[:,i])
                decoder_input = ref[:,i:i+1]

        else:
            for i in range(1, ref_max_len):
                decoder_out, dec_h, attn_weights = decoder(
                        decoder_input, dec_h, encoder_out)
                loss += utils.batch_loss(loss_fn, decoder_out, ref[:,i], mask[:,i])

                _, topi = decoder_out.data.topk(1)
                decoder_input = Variable(topi)

        total_loss += np.asscalar(loss.data.cpu().numpy())
        loss /= float(len(ref_len))
        batches += 1
        loss.backward()

        clip_grad_norm(encoder.parameters(), conf.grad_clip)
        clip_grad_norm(decoder.parameters(), conf.grad_clip)

        enc_opt.step()
        dec_opt.step()

        if batches % 100 == 0:
            print('minibatch #{:4d}, average loss: {:4.4f}'.format(
                batches, total_loss / batches / batch_size))

    return total_loss / len(dataloader.dataset)


def main():
    data_folder = sys.argv[1]
    src_lang = sys.argv[2]
    ref_lang = sys.argv[3]
    conf = RNNConfig(data_folder, src_lang, ref_lang)

    src_vocab = pickle.load(open(sys.argv[4], 'rb'))
    ref_vocab = pickle.load(open(sys.argv[5], 'rb'))

    train_dataset = NMTDataset(
            load_data(conf.train_src_path), 
            load_data(conf.train_ref_path), 
            src_vocab, 
            ref_vocab)
    train_dataloader = NMTDataLoader(
            train_dataset, 
            batch_size=conf.batch_size, 
            num_workers=0, 
            shuffle=True)
    print('%d training dataset loaded.' % len(train_dataset))
    
    dev_dataset = NMTDataset(
            load_data(conf.dev_src_path), 
            load_data(conf.dev_ref_path), 
            src_vocab, 
            ref_vocab)
    dev_dataloader = NMTDataLoader(
            dev_dataset, 
            batch_size=conf.batch_size, 
            num_workers=0)
    print('%d validation dataset loaded.' % len(dev_dataset))

    save_name = conf.save_path+'/%srnn' % ('w' if conf.word_level else '')
    start = 0
    if os.path.exists(save_name+'_encoder_'+str(start-1)):
        encoder = torch.load(save_name+'_encoder_'+str(start-1))
        decoder = torch.load(save_name+'_decoder_'+str(start-1))
    else:
        encoder = RNNEncoder(
                conf.encoder_emb_size, 
                conf.hid_dim, 
                conf.vocab_sizes, 
                train_dataset.src_vocab, 
                conf.encoder_layers, 
                conf.encoder_dropout, 
                conf.word_level)
        decoder = RNNDecoder(
                conf.decoder_emb_size, 
                conf.hid_dim, 
                len(train_dataset.ref_vocab), 
                conf.decoder_layers, 
                conf.decoder_dropout)

    if conf.cuda:
        encoder.cuda()
        decoder.cuda()

    best_bleu = -1
    best_encoder = copy.deepcopy(encoder)
    best_decoder = copy.deepcopy(decoder)
    for epoch in range(start, start + conf.epochs):
        print('Epoch [{:3d}]'.format(epoch))
        train_loss = train(encoder, decoder, train_dataloader, conf)
        print('Training loss:\t%f' % train_loss)

        bleus = 0
        for _, (src, ref, cand, bleu) in enumerate(
                evaluate_rnn(encoder, decoder, dev_dataloader, conf.beam)):
            bleus += sum(bleu)
        bleus /= len(dev_dataloader.dataset)

        print('Avg BLEU score:{:8.4f}'.format(bleus))

        if bleus > best_bleu:
            #best_bleu = bleus
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)
            torch.save(best_encoder.cpu(), save_name+'_encoder_'+str(epoch))
            torch.save(best_decoder.cpu(), save_name+'_decoder_'+str(epoch))

    #torch.save(best_encoder.cpu(), save_name+'_encoder')
    #torch.save(best_decoder.cpu(), save_name+'_decoder')


if __name__ == '__main__':
    main()
