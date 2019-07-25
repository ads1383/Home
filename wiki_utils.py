import os
import torch
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext import vocab
import spacy
from spacy.symbols import ORTH
from torchtext.datasets import WikiText2

my_tok = spacy.load('en')

def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]


def WikiTexts(batch_size = 32, bptt = 30, vectors="glove.6B.100d"): 
    my_tok = spacy.load('en')
    #my_tok.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
    #my_tok.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
    #my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
    TEXT = data.Field(lower=True, tokenize=spacy_tok)
    train, valid, test = WikiText2.splits(TEXT)
    TEXT.build_vocab(train, vectors=vectors)
    train_loader, val_loader, test_loader = data.BPTTIterator.splits(
                                                        (train, valid, test),
                                                         batch_size=batch_size,
                                                         bptt_len=bptt, # this is where we specify the sequence length
                                                         #device=(0 if USE_GPU else -1),
                                                         repeat=False)
    
    return train_loader, val_loader, test_loader, TEXT







