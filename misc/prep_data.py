
from misc.lang import Lang

import unicodedata
import re
from io import open


def prepare_data(lang1, lang2, reverse=False):
    """
    (1) Read text file and split into lines, split lines into pairs
    (2) Normalize text, filter by length and content
    (3) Make word lists from sentences in pairs
    """
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Reading {} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words : ")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    src_lines = open('data/dev/comb.src', encoding='utf-8').read().strip().split('\n')
    tgt_lines = open('data/dev/comb.tgt', encoding='utf-8').read().strip().split('\n')

    pairs = [[normalize_str(s1), normalize_str(s2)] for s1, s2 in zip(src_lines, tgt_lines)]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


MAX_LENGTH = 20  # 10


def filter_pair(p, max_len=MAX_LENGTH):
    return len(p[0].split(" ")) < max_len and \
        len(p[1].split(" ")) < max_len


def normalize_str(s):
    """lowercase, trim, remove non-letter chars"""
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def unicode2ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
