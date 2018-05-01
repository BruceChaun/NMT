# Neural Machine Translation

This is a PyTorch implementation of neural machine translation.

## Requirements

* PyTorch 0.2/0.3
* Python 3.5/3.6
* NumPy
* NLTK

## Models

1. The recurrent model is [1]. The encoder is a two-layer bidirectinoal GRU and the decoder is a single-layer unidirectional GRU. The attention is computed by a multi-layer perceptron.

2. The ConvS2S is proposed by [2], which uses entirely convolutional layers to encode and decode.

3. The Transformer is proposed by [3], which uses multi-head attention and position-wise feed-forward network in both encoder and decoder.

4. Hashed character n-gram embedding [4], which represents word by its character n-grams. Other procedures are the same.

## Usage

1. Download dataset from [WIT3](https://wit3.fbk.eu/mt.php?release=2016-01).

2. Preprocess data
```bash
python preprocess.py data/en-de/train.tags.en-de.de data/en-de/train.tags.en-de.en \
    data/en-de/de.train data/en-de/en.train # parse raw data
python preprocess.py data/en-de/de.train data/en-de/de.vocab.p 3 # build source vocab
python preprocess.py data/en-de/en.train data/en-de/en.vocab.p 3 # build target vocab
```

3. Train models
```bash
python [train_rnn.py|train_cnn.py|train_attn.py] \
    data/en-de de en data/en-de/de.vocab.p data/en-de/en.vocab.p
```

4. Evaluate models
```bash
python eval.py data/en-de de en [rnn|cnn|attn] \
    data/en-de/de.vocab.p data/en-de/en.vocab.p ckpt/encoder ckpt/decoder
```

## Results

### Model Complexity

| Model | #params(word/n-gram) |
|:-----:|:--------------------:|
| RNN | 83,941,203 / 64,440,659 |
| ConvS2S | 74,432,474 / 54,931,930 |
| Transformer | 26,960,896 / 17,771,264 |

### BLEU score

| Model | beam=1 | beam=3 | beam=5 |
|:-----:|:------:|:------:|:------:|
| RNN(word) | 7.3 | 8.6 | 8.8 |
| RNN(n-gram) | 4.0 | 4.6 | 4.8 |
| ConvS2S(word) | 14.2 | 17.2 | 17.8 |
| ConvS2S(n-gram) | 11.0 | 13.7 | 14.4 |
| Transformer(word) | 4.0 | 4.1 | 4.2 |
| Transformer(n-gram) | 4.1 | 4.3 | 4.2 |

### Inference Speed

We test the inference efficiency with beam size 3 on Tesla K80 with 1 GPU card.

| Model | #speed(sec)(word/n-gram) |
|:-----:|:--------------------:|
| RNN | 0.36 / 0.39 |
| ConvS2S | 1.64 / 1.68 |
| Transformer | 0.49 / 0.51 |

## References

1. [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)
2. [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
4. [Natural language processing with small feed-forward networks](https://arxiv.org/abs/1708.00214)
