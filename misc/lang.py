
import torch
from net.enc_dec import DEV_


def sentence2idxes(lang, sentence):
    return [lang.word2idx[word] for word in sentence.split(' ')]


def sentence2tensor(lang, sentence, dev=DEV_):
    indexes = sentence2idxes(lang, sentence)
    indexes.append(Lang.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=dev).view(-1, 1)


def pair2tensors(pair, input_lang, output_lang):
    input_tensor = sentence2tensor(input_lang, pair[0])
    target_tensor = sentence2tensor(output_lang, pair[1])
    return (input_tensor, target_tensor)


class Lang:

    SOS_token = 0  # start of sentence
    EOS_token = 1  # end of sentence

    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.idx2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # SOS & EOS

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
