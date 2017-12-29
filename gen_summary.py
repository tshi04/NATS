import re
import os
import argparse
import shutil
import glob
import time

import torch
from torch.autograd import Variable

from data_utils import *
from utils import *

data_dir = '../sum_data/'
file_vocab = 'vocab'
file_test = 'test.txt'
batch_size = 8

vocab2id, id2vocab = construct_vocab(
    file_=data_dir+'/'+file_vocab,
    max_size=50000,
    mincount=5
)
print 'The vocabulary size: {0}'.format(len(vocab2id))
# test data
test_batch = create_batch_file(
    path_=data_dir,
    fkey_='test',
    file_=file_test,
    batch_size=batch_size,
    clean=False
)
print 'The number of batches (test): {0}'.format(test_batch)

model = torch.load('../sum_data/seq2seq_results-0/seq2seq_15_1000.pt').cuda()
print model

start_time = time.time()
fout = open(data_dir+'/summaries.txt', 'w')
for batch_id in range(test_batch):
    
    src_var, trg_input_var, trg_output_var = process_minibatch(
        batch_id=batch_id, path_=data_dir, fkey_='test', 
        batch_size=batch_size, vocab2id=vocab2id, 
        max_lens=[400, 120]
    )
    beam_seq, beam_prb = batch_beam_search(model=model, src_text=src_var, vocab2id=vocab2id, max_len=100)
    trg_seq = trg_output_var.data.numpy()
    for b in range(batch_size):
        arr = []
        gen_text = beam_seq.data.cpu().numpy()[b,0]
        gen_text = [id2vocab[wd] for wd in gen_text]
        arr.append(' '.join(gen_text))
        trg_text = [id2vocab[wd] for wd in trg_seq[b]]
        arr.append(' '.join(trg_text))
        fout.write('<sec>'.join(arr)+'\n')

    end_time = time.time()
    print(batch_id, end_time-start_time)
    
fout.close()