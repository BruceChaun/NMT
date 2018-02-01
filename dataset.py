import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from collections import Counter
import numpy as np
from functools import partial


class NMTDataset(Dataset):

    def __init__(self, src, ref, src_vocab=None, ref_vocab=None):
        self.src, self.src_vocab = self._convert(src, src_vocab)
        self.ref, self.ref_vocab = self._convert(ref, ref_vocab)

        assert len(self.src) == len(self.ref), \
                'sentence pairs must be in the same length.'

        self.src = self._corpus2idx(self.src, self.src_vocab)
        self.ref = self._corpus2idx(self.ref, self.ref_vocab)


    def _vocab_from_corpus(self, corpus):
        vocab = Counter()
        for sent in corpus:
            vocab.update(sent)
        vocab = {w:i+4 for i, w in enumerate(vocab)}
        vocab['<PAD>'] = 0
        vocab['<SOS>'] = 1
        vocab['<EOS>'] = 2
        vocab['<OOV>'] = 3
        return vocab

    
    def _convert(self, corpus, vocab):
        if vocab is None:
            corpus = [['<SOS>'] + sent + ['<EOS>'] for sent in corpus]
            return corpus, self._vocab_from_corpus(corpus)

        new_vocab = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<OOV>':3}
        for i, w in enumerate(vocab):
            new_vocab[w] = i + 4

        converted = []
        for sent in corpus:
            converted.append([])
            for w in ['<SOS>'] + sent + ['<EOS>']:
                if w in new_vocab:
                    converted[-1].append(w)
                else:
                    converted[-1].append('<OOV>')
        return converted, new_vocab


    def _corpus2idx(self, corpus, vocab):
        corpus_idx = []
        for sent in corpus:
            corpus_idx.append([vocab[w] for w in sent])
        return corpus_idx


    def __len__(self):
        return len(self.src)


    def __getitem__(self, i):
        return self.src[i], self.ref[i]


def collate(samples):
    def _pad_idx(seq):
        lens = [len(s) for s in seq]
        max_len = max(lens)
        for i in range(len(seq)):
            seq[i] = np.pad(seq[i], (0, max_len-len(seq[i])), 
                    'constant', constant_values=0)
        return np.array(seq)

    src, ref = [list(x) for x in zip(*samples)]
    src = _pad_idx(src)
    ref = _pad_idx(ref)
    
    return Variable(torch.LongTensor(src)), Variable(torch.LongTensor(ref))


NMTDataLoader = partial(
        DataLoader, 
        shuffle=False, 
        collate_fn=collate, 
        drop_last=False)


def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(line.strip().split())
    return data


if __name__ == '__main__':
    ''' test only '''
    import sys
    import pickle

    src = load_data(sys.argv[1])
    ref = load_data(sys.argv[2])
    vocab = pickle.load(open(sys.argv[3], 'rb'))
    vocab = {w:i for i, w in enumerate(vocab)}

    dataset = NMTDataset(src, ref, None, vocab)
    dataloader = NMTDataLoader(dataset, batch_size=8, num_workers=0, shuffle=True)

    src_ivocab = {i:w for w,i in dataset.src_vocab.items()}
    ref_ivocab = {i:w for w,i in dataset.ref_vocab.items()}
    for src, ref in dataloader:
        print(len(src), len(ref))
        src = src.data.numpy()
        ref = ref.data.numpy()
        for i in range(len(src)):
            print(len(src[i]), len(ref[i]))
            print(' '.join([src_ivocab[j] for j in src[i]]))
            print(' '.join([ref_ivocab[j] for j in ref[i]]))
        break
