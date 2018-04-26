import torch
import numpy as np
import pickle
import sys


def main():
    word_encoder = torch.load(sys.argv[1])
    word_embedding = word_encoder.embedding.weight.data.numpy()[4:]
    print('embedding dim: ', word_embedding.shape)
    np.savetxt('word_embedding.tsv', word_embedding, fmt='%1.4f', delimiter='\t')
    print('word level done!')

    encoder = torch.load(sys.argv[2])
    embedding = encoder.cuda().embedding
    vocab = pickle.load(open(sys.argv[3], 'rb'))
    print('vocab size: %d' % len(vocab))
    vocab = {w:i+4 for w,i in vocab.items()}
    embedding_array = np.zeros([len(vocab), embedding.emb_size * len(embedding.vocab_sizes)])
    f = open('vocab.tsv', 'w')
    for k, i in vocab.items():
        f.write(k+'\n')
        embedding_array[i-4] = embedding._word_embedding(k).data.cpu().numpy()
    np.savetxt('embedding.tsv', embedding_array, fmt='%1.4f', delimiter='\t')
    f.close()
    print('done')


if __name__ == '__main__':
    main()
