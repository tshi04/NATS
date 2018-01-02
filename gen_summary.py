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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../sum_data', help='directory that store the data.')
parser.add_argument('--model_dir', default='seq2seq_results-0', help='directory that store the model.')
parser.add_argument('--model_file', default='seq2seq_0_0.pt', help='file for model.')
parser.add_argument('--file_vocab', default='vocab', help='file store training vocabulary.')
parser.add_argument('--file_test', default='test.txt', help='test data')
parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
parser.add_argument('--vocab_size', type=int, default=50000, help='max number of words in the vocabulary.')
parser.add_argument('--word_mincount', type=int, default=5, 
                    help='min count of the words in the corpus in the vocab')
parser.add_argument('--src_seq_lens', type=int, default=400, help='length of source documents.')
parser.add_argument('--trg_seq_lens', type=int, default=120, help='length of trage documents.')
opt = parser.parse_args()

vocab2id, id2vocab = construct_vocab(
    file_=os.path.join(opt.data_dir, opt.file_vocab),
    max_size=opt.vocab_size,
    mincount=opt.word_mincount
)
print 'The vocabulary size: {0}'.format(len(vocab2id))
# test data
test_batch = create_batch_file(
    path_=opt.data_dir,
    fkey_='test',
    file_=opt.file_test,
    batch_size=opt.batch_size,
    clean=False
)
print 'The number of batches (test): {0}'.format(test_batch)

model = torch.load(os.path.join(opt.data_dir, opt.model_dir, opt.model_file)).cuda()
print model

start_time = time.time()
fout = open(os.path.join(opt.data_dir, 'summaries.txt'), 'w')
for batch_id in range(test_batch):
    
    src_var, trg_input_var, trg_output_var = process_minibatch(
        batch_id=batch_id, path_=opt.data_dir, fkey_='test', 
        batch_size=opt.batch_size, vocab2id=vocab2id, 
        max_lens=[opt.src_seq_lens, opt.trg_seq_lens]
    )
    beam_seq, beam_prb = batch_beam_search(model=model, src_text=src_var, vocab2id=vocab2id, max_len=opt.trg_seq_lens)
    trg_seq = trg_output_var.data.numpy()
    for b in range(opt.batch_size):
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

rouge_path = os.path.join(opt.data_dir, 'rouge')
sys_smm_path = os.path.join(rouge_path, 'system_summaries')
mod_smm_path = os.path.join(rouge_path, 'model_summaries')
shutil.rmtree(rouge_path)
os.makedirs(rouge_path)
os.makedirs(sys_smm_path)
os.makedirs(mod_smm_path)
fp = open(os.path.join(opt.data_dir, 'summaries.txt'), 'r')
cnt = 1
for line in fp:
    arr = re.split('<sec>', line[:-1])
    smm = re.split('<pad>|<s>|</s>', arr[0])
    rmm = re.split('<pad>|<s>|</s>', arr[1])
    rmm = filter(None, rmm)
    smm = filter(None, smm)
    rmm = [' '.join(filter(None, re.split('\s', sen))) for sen in rmm]
    smm = [' '.join(filter(None, re.split('\s', sen))) for sen in smm]
    rmm = filter(None, rmm)
    smm = filter(None, smm)[:3]
    fout = open(os.path.join(sys_smm_path, 'sum.'+str(cnt).zfill(5)+'.txt'), 'w')
    for sen in rmm:
        arr = re.split('\s', sen)
        arr = ['[unk]' if wd == '<unk>' else wd for wd in arr]
        dstr = ' '.join(arr)
        fout.write(dstr+'\n')
    fout.close()
    fout = open(os.path.join(mod_smm_path, 'sum.A.'+str(cnt).zfill(5)+'.txt'), 'w')
    for sen in smm:
        arr = re.split('\s', sen)
        arr = ['[unk]' if wd == '<unk>' else wd for wd in arr]
        dstr = ' '.join(arr)
        fout.write(dstr+'\n')
    fout.close()
    cnt += 1
fp.close()