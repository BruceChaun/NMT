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

        self.n_grams = 4


class RNNConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)


class CNNConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)


class AttnConfig(Config):

    def __init__(self, data_folder, src_lang, ref_lang):
        Config.__init__(self, data_folder, src_lang, ref_lang)
