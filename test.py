import re
import os
import argparse
import shutil
import glob

import torch
from torch.autograd import Variable

from model import *
from data_utils import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../sum_data/', help='directory that store the data')
parser.add_argument('--file_vocab', default='cnn_vocab.txt', help='vocabulary file')
parser.add_argument('--file_corpus', default='cnn.txt', help='file store documents')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--src_emb_dim', type=int, default=100, help='source embedding dimension')
parser.add_argument('--trg_emb_dim', type=int, default=100, help='target embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=100, help='encoder hidden dimension')
parser.add_argument('--trg_hidden_dim', type=int, default=100, help='decoder hidden dimension')
parser.add_argument('--src_num_layers', type=int, default=2, help='encoder number layers')
parser.add_argument('--trg_num_layers', type=int, default=1, help='decoder number layers')
parser.add_argument('--src_bidirection', type=bool, default=True)
parser.add_argument('--batch_first', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--attn_method', default='bahdanau', help='vanilla | bahdanau | luong_dot | luong_concat | luong_general')
parser.add_argument('--network_', default='gru')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--src_max_lens', type=int, default=256)
parser.add_argument('--trg_max_lens', type=int, default=30)
opt = parser.parse_args()

model = torch.load('../sum_data/seq2seq_results-1/seq2seq_299_1000.pt').cuda()
print model

data_dir = '../sum_data/'
file_vocab = 'cnn_vocab_abs.txt'
file_corpus = 'cnn_abs.txt'
batch_size = 64

vocab2id, id2vocab = construct_vocab(data_dir+'/'+file_vocab)
print 'The vocabulary size: {0}'.format(len(vocab2id))

if not os.path.exists('batch_folder'):
    n_batch = create_batch_file(
        file_name=data_dir+'/'+file_corpus, 
        batch_size=batch_size
    )
else:
    n_batch = len(glob.glob('batch_folder/batch_*'))
print 'The number of batches: {0}'.format(n_batch)

src_var, trg_input_var, trg_output_var = process_minibatch(
    200, vocab2id, max_lens=[100, 20])
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
