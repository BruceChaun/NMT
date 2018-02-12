import torch
from torch.autograd import Variable
import numpy as np
import pickle
import sys
import os
import copy

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
        decoder_input = src[:,:1]
        src = src[:,1:] # skip <SOS>
        batch_size, src_max_len = src.size()
        enc_h = encoder.init_hidden(batch_size)

        if cuda:
            src = src.cuda()
            enc_h = enc_h.cuda()
            decoder_input = decoder_input.cuda()

        encoder_out, enc_h = encoder(src, enc_h, src_len)
        dec_h = enc_h[-2:-1] # last forward layer in encoder

        '''
        # greedy search
        candidate = np.zeros([batch_size, max_len], dtype=int)
        for i in range(max_len):
            decoder_out, dec_h, attn_weights = decoder(
                    decoder_input, dec_h, encoder_out)
            _, decoder_input = decoder_out.data.topk(1)
            candidate[:,i:i+1] = decoder_input.cpu().numpy()
            decoder_input = Variable(decoder_input)
        '''

        # beam search
        candidate = torch.zeros([batch_size, beam, max_len]).long()

        decoder_input = Variable(torch.ones([batch_size, beam, 1]).long())
        prob_matrix = torch.zeros([batch_size, beam])
        dec_hs = torch.cat([dec_h.unsqueeze(0)] * beam)
        if cuda:
            prob_matrix = prob_matrix.cuda()
            dec_hs = dec_hs.cuda()

        for i in range(max_len):
            tmp_prob = torch.zeros([batch_size, beam**2])
            indices = torch.zeros([batch_size, beam**2])
            if cuda:
                decoder_input = decoder_input.cuda()
                tmp_prob = tmp_prob.cuda()
                indices = indices.cuda()

            for b in range(beam):
                decoder_out, dec_hs[b], _ = decoder(
                        decoder_input[:,b,:], dec_hs[b], encoder_out)
                topv, topi = decoder_out.data.topk(beam)
                tmp_prob[:,b*beam:(b+1)*beam] = prob_matrix + topv
                indices[:,b*beam:(b+1)*beam] = topi

            prob_matrix, topi = tmp_prob.topk(beam)
            for batch in range(batch_size):
                candidate[batch,:,i] = indices[batch].index_select(0, topi[batch])
            decoder_input = Variable(candidate[:,:,i:i+1])

        candidate = candidate[:,0,:]

        src_seqs = []
        ref_seqs = []
        cand_seqs = []
        bleus = []
        src = src.data.cpu().numpy()
        for i in range(batch_size):
            src_seq = utils.convert2seq(src[i], src_vocab)
            ref_seq = utils.convert2seq(ref[i,1:], ref_vocab)
            cand_seq = utils.convert2seq(candidate[i], ref_vocab)
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

        if cuda:
            src = src.cuda()

        encoder_out, e = encoder(src)

        '''
        # greedy search
        candidate = torch.zeros([batch_size, max_len]).long()
        candidate[:,0] = 1
        for i in range(1, max_len):
            decoder_input = Variable(candidate[:,:i])
            decoder_out  = decoder(decoder_input, encoder_out)
            _, topi = decoder_out.data.topk(1)
            candidate[:,i:i+1] = topi
        '''

        # beam search
        candidate = torch.zeros([batch_size, beam, max_len]).long()
        candidate[:,:,0] = 1
        prob_matrix = torch.zeros([batch_size, beam])
        tmp_prob = torch.zeros([batch_size, beam**2])
        indices = torch.zeros([batch_size, beam**2])

        if cuda:
            candidate = candidate.cuda()
            prob_matrix = prob_matrix.cuda()
            tmp_prob = tmp_prob.cuda()
            indices = indices.cuda()

        for i in range(1, max_len):
            tmp_prob.zero_()
            indices.zero_()
            decoder_input = Variable(candidate[:,:,:i])

            for b in range(beam):
                decoder_out = decoder(decoder_input[:,b,:], encoder_out)
                topv, topi = decoder_out.data.topk(beam)
                tmp_prob[:,b*beam:(b+1)*beam] = prob_matrix + topv
                indices[:,b*beam:(b+1)*beam] = topi

            prob_matrix, topi = tmp_prob.topk(beam)
            for batch in range(batch_size):
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
            load_data(conf.tst_src_path), 
            load_data(conf.tst_ref_path), 
            src_vocab, 
            ref_vocab)
    tst_dataloader = NMTDataLoader(
            tst_dataset, 
            batch_size=conf.batch_size, 
            num_workers=0)
    print('%d test dataset loaded.' % len(tst_dataset))

    save_name = os.path.join(conf.save_path, model_type)
    encoder = torch.load(save_name+'_encoder')
    decoder = torch.load(save_name+'_decoder')
    if conf.cuda:
        encoder.cuda()
        decoder.cuda()

    eval_fn = locals()['evaluate_%s' % model_type]
    bleus = 0
    for _, (src, ref, cand, bleu_) in enumerate(
            eval_fn(encoder, decoder, tst_dataloader, conf.beam)):
        bleus += sum(bleu_)
        for i in range(len(src)):
            print('Source\n%s' % src[i])
            print('Reference\n%s' % ref[i])
            print('Translation\n%s' % cand[i])
            print('BLEU\t{:3.4f}'.format(bleu_[i]))

    print('Avg BLEU score:{:8.4f}'.format(bleus/len(tst_dataloader.dataset)))
