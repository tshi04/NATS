import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='seq2seq', help='seq2seq | seqGAN')
stc = parser.parse_args()

data_dir = '../sum_data/'
file_vocab = 'cnn_vocab.txt'
file_corpus = 'cnn.txt'
n_epoch = 1

vocab2id, id2vocab = construct_vocab(data_dir+'/'+file_vocab)
print 'The vocabulary size: {0}'.format(len(vocab2id))

n_batch = create_batch_file(file_name='../sum_data/cnn.txt', batch_size=128)
print 'The number of batches: {0}'.format(n_batch)

for epoch in range(n_epoch):
    for batch_id in range(n_batch):
        src_var, trg_var = process_minibatch(
            batch_id, 
            vocab2id, 
            max_len=[100, 400]
        )
        print src_var
        print trg_var
        break