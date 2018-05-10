import torch
from torch.autograd import Variable
import numpy as np
import pickle
import sys
import os
import copy
import time

from dataset import *
import bleu
import utils
import config


def evaluate_rnn(encoder, decoder, dataloader, beam, max_len=150):
    cuda = torch.cuda.is_available()
    encoder.eval()
    decoder.eval()
    src_vocab = {i:w for w,i in dataloader.dataset.src_vocab.items()}
    ref_vocab = {i:w for w,i in dataloader.dataset.ref_vocab.items()}

    for src, src_len, ref, ref_len in dataloader:
        src = Variable(torch.LongTensor(src), volatile=True)
        src = src[:,1:] # skip <SOS>
        batch_size, src_max_len = src.size()
        max_len = min(max_len, src_max_len * 2)
        enc_h = encoder.init_hidden(batch_size)

        if cuda:
            src = src.cuda()
            enc_h = enc_h.cuda()

        encoder_out, enc_h = encoder(src, enc_h, src_len)
        dec_h = enc_h[-2:-1] # last forward layer in encoder

        # beam search
        candidate = torch.zeros([batch_size, beam, max_len]).long()
        candidate[:,:,0] = 1
        prob_matrix = torch.zeros([batch_size, beam])
        tmp_prob = torch.zeros([batch_size, beam**2])
        indices = torch.zeros([batch_size, beam**2])
        decoder_input = Variable(candidate[:,:,:1])
        if cuda:
            prob_matrix = prob_matrix.cuda()
            tmp_prob = tmp_prob.cuda()
            indices = indices.cuda()
            decoder_input = decoder_input.cuda()

        decoder_out, dec_h, _ = decoder(decoder_input[:,0,:], dec_h, encoder_out)
        dec_hs = dec_h.repeat(beam, 1, 1)
        prob_matrix, candidate[:,:,1] = decoder_out.data.topk(beam)

        for i in range(2, max_len):
            tmp_prob.zero_()
            indices.zero_()
            decoder_input = Variable(candidate[:,:,i-1:i])
            if cuda:
                decoder_input = decoder_input.cuda()

            for b in range(beam):
                decoder_out, dec_hs[b:b+1], _ = decoder(
                    decoder_input[:,b,:], dec_hs[b:b+1], encoder_out)
                topv, topi = decoder_out.data.topk(beam)
                tmp_prob[:,b*beam:(b+1)*beam] = prob_matrix[:,b:b+1] + topv
                indices[:,b*beam:(b+1)*beam] = topi

            prob_matrix, topi = tmp_prob.topk(beam)
            for batch in range(batch_size):
                top_candidate = candidate[batch,:,:i].clone()
                prev_dec_hs = dec_hs[:,batch,:].clone()
                prev = topi / beam
                for b in range(beam):
                    candidate[batch,b,:i] = top_candidate[prev[batch,b]]
                    dec_hs[b,batch] = prev_dec_hs[prev[batch,b]]
                candidate[batch,:,i] = indices[batch].index_select(0, topi[batch])

        candidate = candidate[:,0,:]

        src_seqs = []
        ref_seqs = []
        cand_seqs = []
        bleus = []
        src = src.data.cpu().numpy()
        for i in range(batch_size):
            src_seq = utils.convert2seq(src[i], src_vocab)
            ref_seq = utils.convert2seq(ref[i,1:], ref_vocab)
            cand_seq = utils.convert2seq(candidate[i,1:], ref_vocab)
            src_seqs.append(' '.join(src_seq))
            ref_seqs.append(' '.join(ref_seq))
            cand_seqs.append(' '.join(cand_seq))
            if len(cand_seq) == 0:
                bleus.append(0)
            else:
                bleus.append(100 * 
                        bleu.compute_bleu([[ref_seq]], [cand_seq])[0])

        yield (src_seqs, ref_seqs, cand_seqs, bleus)


def evaluate_cnn(encoder, decoder, dataloader, beam, max_len=150):
    cuda = torch.cuda.is_available()
    encoder.eval()
    decoder.eval()
    src_vocab = {i:w for w,i in dataloader.dataset.src_vocab.items()}
    ref_vocab = {i:w for w,i in dataloader.dataset.ref_vocab.items()}

    for src, src_len, ref, ref_len in dataloader:
        src = Variable(torch.LongTensor(src), volatile=True)
        batch_size, src_max_len = src.size()
        max_len = min(max_len, src_max_len * 2)

        if cuda:
            src = src.cuda()

        encoder_out, e = encoder(src)

        # beam search
        candidate = torch.zeros([batch_size, beam, max_len]).long()
        candidate[:,:,0] = 1
        prob_matrix = torch.zeros([batch_size, beam])
        tmp_prob = torch.zeros([batch_size, beam**2])
        indices = torch.zeros([batch_size, beam**2])
        decoder_input = Variable(candidate[:,:,:1])

        if cuda:
            candidate = candidate.cuda()
            prob_matrix = prob_matrix.cuda()
            tmp_prob = tmp_prob.cuda()
            indices = indices.cuda()
            decoder_input = decoder_input.cuda()

        decoder_out = decoder(decoder_input[:,0,:], encoder_out)
        prob_matrix, candidate[:,:,1] = decoder_out.data.topk(beam)

        for i in range(2, max_len):
            tmp_prob.zero_()
            indices.zero_()
            decoder_input = Variable(candidate[:,:,:i])
            if cuda:
                decoder_input = decoder_input.cuda()

            for b in range(beam):
                decoder_out = decoder(decoder_input[:,b,:], encoder_out)
                topv, topi = decoder_out.data.topk(beam)
                tmp_prob[:,b*beam:(b+1)*beam] = prob_matrix[:,b:b+1] + topv
                indices[:,b*beam:(b+1)*beam] = topi

            prob_matrix, topi = tmp_prob.topk(beam)
            for batch in range(batch_size):
                top_candidate = candidate[batch,:,:i].clone()
                prev = topi / beam
                for b in range(beam):
                    candidate[batch,b,:i] = top_candidate[prev[batch,b], :]
                candidate[batch,:,i] = indices[batch].index_select(0, topi[batch])

        candidate = candidate[:,0,:]

        src_seqs = []
        ref_seqs = []
        cand_seqs = []
        bleus = []
        src = src.data.cpu().numpy()
        for i in range(batch_size):
            src_seq = utils.convert2seq(src[i,1:], src_vocab)
            ref_seq = utils.convert2seq(ref[i,1:], ref_vocab)
            cand_seq = utils.convert2seq(candidate[i,1:], ref_vocab)
            src_seqs.append(' '.join(src_seq))
            ref_seqs.append(' '.join(ref_seq))
            cand_seqs.append(' '.join(cand_seq))
            if len(cand_seq) == 0:
                bleus.append(0)
            else:
                bleus.append(100 * 
                        bleu.compute_bleu([[ref_seq]], [cand_seq])[0])

        del src, candidate, prob_matrix, tmp_prob, indices
        torch.cuda.empty_cache()

        yield (src_seqs, ref_seqs, cand_seqs, bleus)


def evaluate_attn(encoder, decoder, dataloader, beam, max_len=75):
    cuda = torch.cuda.is_available()
    encoder.eval()
    decoder.eval()
    src_vocab = {i:w for w,i in dataloader.dataset.src_vocab.items()}
    ref_vocab = {i:w for w,i in dataloader.dataset.ref_vocab.items()}

    for src, src_len, ref, ref_len in dataloader:
        src = Variable(torch.LongTensor(src), volatile=True)
        batch_size, src_max_len = src.size()
        max_len = min(max_len, src_max_len * 2)

        if cuda:
            src = src.cuda()

        encoder_out = encoder(src)

        # beam search
        candidate = torch.zeros([batch_size, beam, max_len]).long()
        candidate[:,:,0] = 1
        prob_matrix = torch.zeros([batch_size, beam])
        tmp_prob = torch.zeros([batch_size, beam**2])
        indices = torch.zeros([batch_size, beam**2])
        decoder_input = Variable(candidate[:,:,:1])

        if cuda:
            candidate = candidate.cuda()
            prob_matrix = prob_matrix.cuda()
            tmp_prob = tmp_prob.cuda()
            indices = indices.cuda()
            decoder_input = decoder_input.cuda()

        decoder_out = decoder(src, decoder_input[:,0,:], encoder_out)
        prob_matrix, candidate[:,:,1] = decoder_out.data.topk(beam)

        for i in range(2, max_len):
            tmp_prob.zero_()
            indices.zero_()
            decoder_input = Variable(candidate[:,:,:i])
            if cuda:
                decoder_input = decoder_input.cuda()

            for b in range(beam):
                decoder_out = decoder(src, decoder_input[:,b,:], encoder_out)
                topv, topi = decoder_out.data.topk(beam)
                tmp_prob[:,b*beam:(b+1)*beam] = prob_matrix[:,b:b+1] + topv
                indices[:,b*beam:(b+1)*beam] = topi

            prob_matrix, topi = tmp_prob.topk(beam)
            for batch in range(batch_size):
                top_candidate = candidate[batch,:,:i].clone()
                prev = topi / beam
                for b in range(beam):
                    candidate[batch,b,:i] = top_candidate[prev[batch,b], :]
                candidate[batch,:,i] = indices[batch].index_select(0, topi[batch])

        candidate = candidate[:,0,:]

        src_seqs = []
        ref_seqs = []
        cand_seqs = []
        bleus = []
        src = src.data.cpu().numpy()
        candidate = candidate.cpu().numpy()
        for i in range(batch_size):
            src_seq = utils.convert2seq(src[i,1:], src_vocab)
            ref_seq = utils.convert2seq(ref[i,1:], ref_vocab)
            cand_seq = utils.convert2seq(candidate[i,1:], ref_vocab)
            src_seqs.append(' '.join(src_seq))
            ref_seqs.append(' '.join(ref_seq))
            cand_seqs.append(' '.join(cand_seq))
            if len(cand_seq) == 0:
                bleus.append(0)
            else:
                bleus.append(100 * 
                        bleu.compute_bleu([[ref_seq]], [cand_seq])[0])

        yield (src_seqs, ref_seqs, cand_seqs, bleus)


if __name__ == '__main__':
    data_folder = sys.argv[1]
    src_lang = sys.argv[2]
    ref_lang = sys.argv[3]
    model_type = sys.argv[4]
    conf = getattr(config, model_type.upper()+'Config')(data_folder, src_lang, ref_lang)

    src_vocab = pickle.load(open(sys.argv[5], 'rb'))
    ref_vocab = pickle.load(open(sys.argv[6], 'rb'))

    tst_dataset = NMTDataset(
            load_data(conf.dev_src_path), 
            load_data(conf.dev_ref_path), 
            src_vocab, 
            ref_vocab)
    tst_dataloader = NMTDataLoader(
            tst_dataset, 
            batch_size=conf.batch_size, 
            num_workers=0)
    print('%d test dataset loaded.' % len(tst_dataset))

    encoder = torch.load(sys.argv[7])
    decoder = torch.load(sys.argv[8])
    encoder_size = utils.model_size(encoder)
    decoder_size = utils.model_size(decoder)
    total_size = encoder_size + decoder_size
    print('[*] Number of model parameters: \nencoder -- {:,}, \
            decoder -- {:,}\ntotal size -- {:,}'.format(
                encoder_size, decoder_size, total_size))
    if conf.cuda:
        encoder.cuda()
        decoder.cuda()

    eval_fn = locals()['evaluate_%s' % model_type]
    start_time = time.time()
    print('Evaluating ...')
    bleus = 0
    for _, (src, ref, cand, bleu_) in enumerate(
            eval_fn(encoder, decoder, tst_dataloader, conf.beam)):
        bleus += sum(bleu_)
        #for i in range(len(src)):
        #    print('Source\n%s' % src[i])
        #    print('Reference\n%s' % ref[i])
        #    print('Translation\n%s' % cand[i])
        #    print('BLEU\t{:3.4f}'.format(bleu_[i]))

    end_time = time.time()
    duration = end_time - start_time
    print('Avg BLEU score:{:8.4f}'.format(bleus/len(tst_dataset)))
    print('Total time elapsed -- {:4.2f}, speed -- {:2.2f}'.format(
        duration, duration / len(tst_dataset)))

