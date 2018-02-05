import re
import sys
import pickle

from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
from collections import Counter


def parse_train(in_filename1, in_filename2, out_filename1, out_filename2, max_len=100):
    out_file1 = open(out_filename1, 'w')
    out_file2 = open(out_filename2, 'w')

    with open(in_filename1, 'r') as f1:
        with open(in_filename2, 'r') as f2:
            pattern = '<.*?>'
            for line1 in f1:
                line1 = line1.strip().lower()
                line2 = next(f2)
                matched = re.findall(pattern, line1)
                if len(matched) == 0:
                    tokens = word_tokenize(line1)
                    if len(tokens) <= max_len:
                        out_file1.write(' '.join(tokens) + "\n")
                        tokens = word_tokenize(line2.strip().lower())
                        out_file2.write(' '.join(tokens) + "\n")

    out_file1.close()
    out_file2.close()


def tree2text(filename, max_len):
    root = ET.parse(filename).getroot()
    root = root.getchildren()[0]

    texts = []
    for doc in root.iter('doc'):
        for seg in doc.iter('seg'):
            texts.append(seg.text)

    return texts


def parse_xml(in_filename1, in_filename2, out_filename1, out_filename2, max_len=100):
    out_file1 = open(out_filename1, 'a')
    out_file2 = open(out_filename2, 'a')

    texts1 = tree2text(in_filename1, max_len)
    texts2 = tree2text(in_filename2, max_len)
    assert len(texts1) == len(texts2), 'source and reference must have the same size.'

    for i in range(len(texts1)):
        text = texts1[i].strip().lower()
        tokens = word_tokenize(text)
        if len(tokens) <= max_len:
            out_file1.write(' '.join(tokens) + "\n")
            tokens = word_tokenize(texts2[i].strip().lower())
            out_file2.write(' '.join(tokens) + "\n")

    out_file1.close()
    out_file2.close()


def process_raw_data():
    in_filename1 = sys.argv[1]
    in_filename2 = sys.argv[2]
    out_filename1 = sys.argv[3]
    out_filename2 = sys.argv[4]
    if 'train' in in_filename1:
        parse_train(in_filename1, in_filename2, out_filename1, out_filename2)
    else:
        parse_xml(in_filename1, in_filename2, out_filename1, out_filename2)


def build_vocab():
    train_filename = sys.argv[1]
    out_filename = sys.argv[2]
    min_count = int(sys.argv[3])

    vocab = Counter()
    with open(train_filename, 'r') as f:
        for line in f:
            vocab.update(line.strip().split())
    print(len(vocab))

    vocab = [w for w, c in vocab.items() if c >= min_count]
    print(len(vocab))
    vocab = {w:i for i,w in enumerate(vocab)}
    pickle.dump(vocab, open(out_filename, 'wb'))


def build_all_vocab():
    sets = ('.train', '.dev', '.tst')
    lang = sys.argv[1]
    out_filename = sys.argv[2]

    vocab = Counter()
    for suffix in sets:
        with open(lang+suffix, 'r') as f:
            for line in f:
                vocab.update(line.strip().split())
    print(len(vocab))
    vocab = {w:i for i,w in enumerate(vocab)}
    pickle.dump(vocab, open(out_filename, 'wb'))


if __name__ == '__main__':
    process_raw_data()
    build_vocab()
    build_all_vocab()
