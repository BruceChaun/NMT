import os
import torch


class Config(object):

    def __init__(self, data_folder, src_lang, ref_lang):
        self.train_src_path = os.path.join(data_folder, src_lang+'.train')
        self.dev_src_path = os.path.join(data_folder, src_lang+'.dev')
        self.tst_src_path = os.path.join(data_folder, src_lang+'.tst')

        self.train_ref_path = os.path.join(data_folder, ref_lang+'.train')
        self.dev_ref_path = os.path.join(data_folder, ref_lang+'.dev')
        self.tst_ref_path = os.path.join(data_folder, ref_lang+'.tst')

        self.save_path = 'saver'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.cuda = torch.cuda.is_available()

        self.vocab_sizes = [53, 499, 997, 2003]
        self.teaching_ratio = 0.5
        self.epochs = 100
        self.batch_size = 64
        self.beam = 1
        self.word_level = False
        self.grad_clip = 1
        self.lr = 0.001


class RNNConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)

        self.encoder_emb_size = 512 if self.word_level else 50
        self.decoder_emb_size = 512 if self.word_level else 300
        self.hid_dim = 256
        self.encoder_layers = 2
        self.decoder_layers = 1
        self.encoder_dropout = 0.1
        self.decoder_dropout = 0.1


class CNNConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)

        self.encoder_emb_size = 512 if self.word_level else 64
        self.decoder_emb_size = 512 if self.word_level else 256
        self.encoder_kernels = [3] * 9
        self.decoder_kernels = [5] * 5
        self.encoder_dropout = 0.1
        self.decoder_dropout = 0.1
        self.batch_size = 8


class AttnConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)

