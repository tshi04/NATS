import re
import os
import argparse
import shutil
import glob

import torch
from torch.autograd import Variable

from data_utils import *
from utils import *

data_dir = '../sum_data/'
file_vocab = 'vocab'
file_test = 'cnn_abs.txt'
batch_size = 64

vocab2id, id2vocab = construct_vocab(
    file_=opt.data_dir+'/'+opt.file_vocab,
    max_size=50000,
    mincount=5
)
print 'The vocabulary size: {0}'.format(len(vocab2id))

model = torch.load('../sum_data/seq2seq_results-0/seq2seq_5_1000.pt').cuda()
print model

src_var, trg_input_var, trg_output_var = process_minibatch(
    500, vocab2id, max_lens=[100, 20])
for k in range(batch_size):
    print
    print
    print ' '.join([id2vocab[wd] for wd in src_var[k].data.cpu().numpy()])
    print
    print ' '.join([id2vocab[wd] for wd in trg_input_var[k].data.cpu().numpy()])
    print
    beam_seq, beam_prb = beam_search(model=model, src_text=src_var[k], vocab2id=vocab2id, max_len=20)
    gen_text = beam_seq.data.cpu().numpy()[0]
    gen_text = [id2vocab[wd] for wd in gen_text]
    print ' '.join(gen_text)
    print '-'*50
