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

from config import CNNConfig
from dataset import *
from encoders import CNNEncoder
from decoders import CNNDecoder
from eval import evaluate_cnn
import utils


def train(encoder, decoder, dataloader, conf):
    encoder.train()
    decoder.train()
    enc_opt = optim.Adam(encoder.parameters(), lr=conf.lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=conf.lr)
    loss_fn = nn.NLLLoss()

    total_loss = 0
    n_batch = 0
    ooms = 0
    for src, src_len, ref, ref_len in dataloader:
        n_batch += 1
        try:
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss = 0

            batch_size, src_max_len = src.shape
            src = Variable(torch.LongTensor(src))
            ref = Variable(torch.LongTensor(ref))

            ref_max_len = ref.size(1)

            if conf.cuda:
                src = src.cuda()
                ref = ref.cuda()

            encoder_out, e = encoder(src)

            decoder_input = ref[:,:1]
            mask = torch.Tensor(utils.mask_matrix(ref_len, ref_max_len))
            teacher_forcing = random.random() < conf.teaching_ratio

            if teacher_forcing:
                for i in range(1, ref_max_len):
                    decoder_out = decoder(decoder_input, encoder_out)
                    loss += utils.batch_loss(loss_fn, decoder_out, ref[:,i], mask[:,i])
                    decoder_input = ref[:,:i+1]

            else:
                for i in range(1, ref_max_len):
                    decoder_out = decoder(decoder_input, encoder_out)
                    loss += utils.batch_loss(loss_fn, decoder_out, ref[:,i], mask[:,i])

                    _, topi = decoder_out.data.topk(1)
                    decoder_input = torch.cat([decoder_input, Variable(topi)], dim=1)

            _loss = np.asscalar(loss.data.cpu().numpy())
            loss /= float(len(ref_len))
            loss.backward()

            clip_grad_norm(encoder.parameters(), conf.grad_clip)
            clip_grad_norm(decoder.parameters(), conf.grad_clip)

            enc_opt.step()
            dec_opt.step()

            total_loss += _loss
            if n_batch % 1000 == 0:
                print('minibatch #{:4d}, average loss: {:4.4f}'.format(
                    n_batch, total_loss / (n_batch - ooms) / batch_size))

        except RuntimeError as e:
            if 'out of memory' in str(e):
                ooms += 1
                torch.cuda.empty_cache()
            else:
                raise e

    if ooms > 0:
        print('Hit [%d] times out of memory error' % ooms)
    return total_loss / (len(dataloader.dataset) - ooms * conf.batch_size)


def main():
    data_folder = sys.argv[1]
    src_lang = sys.argv[2]
    ref_lang = sys.argv[3]
    conf = CNNConfig(data_folder, src_lang, ref_lang)

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

    save_name = conf.save_path+'/%scnn' % ('w' if conf.word_level else '')
    start = 0
    if os.path.exists(save_name+'_encoder_'+str(start-1)):
        encoder = torch.load(save_name+'_encoder_'+str(start-1))
        decoder = torch.load(save_name+'_decoder_'+str(start-1))
    else:
        encoder = CNNEncoder(
                conf.encoder_emb_size, 
                conf.vocab_sizes, 
                train_dataset.src_vocab, 
                conf.encoder_kernels, 
                len(conf.decoder_kernels), 
                conf.encoder_dropout, 
                conf.word_level)
        decoder = CNNDecoder(
                conf.decoder_emb_size, 
                len(train_dataset.ref_vocab), 
                conf.decoder_kernels, 
                conf.decoder_dropout)

    if conf.cuda:
        encoder.cuda()
        decoder.cuda()

    best_bleu = -1
    for epoch in range(start, start + conf.epochs):
        print('Epoch [{:3d}]'.format(epoch))
        train_loss = train(encoder, decoder, train_dataloader, conf)
        print('Training loss:\t%f' % train_loss)

        bleus = 0
        for _, (src, ref, cand, bleu) in enumerate(
                evaluate_cnn(encoder, decoder, dev_dataloader, conf.beam)):
            bleus += sum(bleu)
        bleus /= len(dev_dataloader.dataset)

        print('Avg BLEU score:{:8.4f}'.format(bleus))

        if bleus > best_bleu:
            #best_bleu = bleus
            torch.save(encoder.cpu(), save_name+'_encoder_'+str(epoch))
            torch.save(decoder.cpu(), save_name+'_decoder_'+str(epoch))
            if conf.cuda:
                encoder.cuda()
                decoder.cuda()


if __name__ == '__main__':
    main()
