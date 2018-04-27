import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity


def main():
    vocab = pickle.load(open(sys.argv[1], 'rb'))
    embedding = np.loadtxt(sys.argv[2], delimiter='\t')
    nouns = ['buches', 'kinderbuch', 'kochbuch', 'tagebuch', 'buchladen', 
            'bucht', 'verbuchen']
    adjs = ['gute', 'guter', 'gutem', 'gutes', 'guten', 'zugute']
    verbs = ['komme', 'herkommt', 'kommt', 
            'bekommt', 'hinkommt', 'aufkommt', 'rauskommt']

    def plot_heatmap(words, name):
        word_ids = [vocab[w] for w in words]
        word_vec = np.zeros([len(words), embedding.shape[1]])
        for i, wi in enumerate(word_ids):
            word_vec[i] = embedding[wi]

        ax = sns.heatmap(cosine_similarity(word_vec))
        ax.set_xticklabels(words, rotation=45)
        ax.set_yticklabels(list(reversed(words)), rotation=45)
        plt.savefig(name)
        plt.clf()

    plot_dir = 'plot'
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_heatmap(nouns, os.path.join(plot_dir, 'noun.png'))
    plot_heatmap(adjs, os.path.join(plot_dir, 'adj.png'))
    plot_heatmap(verbs, os.path.join(plot_dir, 'verb.png'))


if __name__ == '__main__':
    main()
