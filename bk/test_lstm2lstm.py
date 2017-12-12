import argparse
import shutil
import re
import os

import torch
from torch.autograd import Variable

from lstm2lstm import *
from data_utils import *

data_dir = '../sum_data/'
file_vocab = 'cnn_vocab.txt'
file_corpus = 'cnn.txt'
n_epoch = 20
batch_size = 64

vocab2id, id2vocab = construct_vocab(data_dir+'/'+file_vocab)
print 'The vocabulary size: {0}'.format(len(vocab2id))

n_batch = create_batch_file(file_name='../sum_data/cnn.txt', batch_size=batch_size)
print 'The number of batches: {0}'.format(n_batch)

model = seq2seq(
    src_emb_dim=100,
    trg_emb_dim=100,
    src_hidden_dim=50,
    trg_hidden_dim=50,
    src_vocab_size=len(vocab2id),
    trg_vocab_size=len(vocab2id),
    src_pad_token=0,
    trg_pad_token=0,
    src_nlayer=2,
    trg_nlayer=1,
    src_bidirect=True,
    batch_size=batch_size,
    dropout=0.0
).cuda()

model.load_state_dict(torch.load('../sum_data/lstm2lstm_results/lstm2lstm_30_1000.model'))

src_var, trg_input_var, trg_output_var = process_minibatch(
    100, vocab2id, max_lens=[256, 24]
)

trg_input_arr = [[vocab2id['<s>'] for k in range(24)] for j in range(batch_size)]
trg_input_var = Variable(torch.LongTensor(trg_input_arr))
for k in range(24):
    print k,
    logits = model(src_var.cuda(), trg_input_var.cuda())
    word_prob = model.decode(logits).data.cpu().numpy().argmax(axis=-1)
    for j in range(64):
        trg_input_arr[j][k] = word_prob[j][k]
        trg_input_var = Variable(torch.LongTensor(trg_input_arr))
print

idid = 32
sen_pred = [id2vocab[x] for x in word_prob[idid]]
print ''.join(['-' for k in range(50)])
st_idx = len(sen_pred)
for k, wd in enumerate(sen_pred):
    if wd == '</s>':
        st_idx = k
        break
sen_pred = sen_pred[:st_idx]
print ' '.join(sen_pred)

print ''.join(['-' for k in range(50)])
sen_abs = [id2vocab[x] for x in trg_output_var.data[idid]]
st_idx = len(sen_abs)
for k, wd in enumerate(sen_abs):
    if wd == '<pad>':
        st_idx = k
        break
print ' '.join(sen_abs[:st_idx])

print ''.join(['-' for k in range(50)])
sen_source = [id2vocab[x] for x in src_var.data[idid]]
st_idx = len(sen_source)
for k, wd in enumerate(sen_source):
    if wd == '<pad>':
        st_idx = k
        break
print ' '.join(sen_source[:st_idx])
