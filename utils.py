import numpy as np


def mask_matrix(seq_len, max_len):
    batches = len(seq_len)
    mask = np.zeros([batches, max_len])

    for i in range(batches):
        mask[i, 1:seq_len[i]] = 1
    return mask


def batch_loss(loss_fn, predict, target, mask):
    assert len(predict) == len(target), \
            'numbers of prediction and target must be the same'

    loss = 0
    for i in range(len(predict)):
        loss += loss_fn(predict[i:i+1], target[i:i+1]) * mask[i]
    return loss


def convert2seq(idx_seq, ivocab):
    seq = []
    for idx in idx_seq:
        if idx == 2:
            break # <EOS> token
        seq.append(ivocab[idx])

    return seq


def model_size(model):
    return sum([p.data.nelement() for p in model.parameters()])
