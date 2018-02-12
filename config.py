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
        self.teaching_ratio = 0.2
        self.epochs = 10
        self.batch_size = 64
        self.beam = 5


class RNNConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)

        self.lr = 0.01
        self.encoder_emb_size = 50
        self.decoder_emb_size = 300
        self.hid_dim = 200
        self.encoder_layers = 2
        self.decoder_layers = 1
        self.encoder_dropout = 0.3
        self.decoder_dropout = 0.3


class CNNConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)

        self.lr = 0.001
        self.encoder_emb_size = 64
        self.decoder_emb_size = 256
        self.encoder_kernels = [3] * 9
        self.decoder_kernels = [5] * 5
        self.encoder_dropout = 0.2
        self.decoder_dropout = 0.2


class AttnConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)

