import os
import re
import glob
import shutil
import random
import numpy as np

import torch
from torch.autograd import Variable
'''
Construct the vocabulary
'''
def construct_vocab(file_, max_size=200000, mincount=5):
    vocab2id = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}
    id2vocab = {2: '<s>', 3: '</s>', 1: '<pad>', 0: '<unk>', 4: '<stop>'}
    word_pad = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}
    
    cnt = 5
    with open(file_, 'r') as fp:
        for line in fp:
            arr = re.split('\s', line[:-1])
            if arr[0] == ' ':
                continue
            if arr[0] in word_pad:
                continue
            if int(arr[1]) >= mincount:
                vocab2id[arr[0]] = cnt
                id2vocab[cnt] = arr[0]
                cnt += 1
            if len(vocab2id) == max_size:
                break
    
    return vocab2id, id2vocab
'''
Split the corpus into batches.
'''
def create_batch_file(path_, fkey_, file_, batch_size):
    file_name = os.path.join(path_, file_)
    folder = os.path.join(path_, 'batch_'+fkey_+'_'+str(batch_size))
    
    try:
        shutil.rmtree(folder)
        os.mkdir(folder)
    except:
        os.mkdir(folder)
    
    corpus_arr = []
    fp = open(file_name, 'r')
    for line in fp:
        corpus_arr.append(line)
    fp.close()
    if fkey_ == 'train':
        random.shuffle(corpus_arr)
        
    cnt = 0
    for itm in corpus_arr:
        try:
            arr.append(itm)
        except:
            arr = [itm]
        if len(arr) == batch_size:
            fout = open(os.path.join(folder, str(cnt)), 'w')
            for sen in arr:
                fout.write(sen)
            fout.close()
            arr = []
            cnt += 1
        
    if len(arr) > 0:
        fout = open(os.path.join(folder, str(cnt)), 'w')
        for sen in arr:
            fout.write(sen)
        fout.close()
        arr = []
        cnt += 1
    
    return cnt
'''
Process the minibatch.
'''
def process_minibatch(batch_id, path_, fkey_, batch_size, src_vocab2id, vocab2id, max_lens=[400, 100]):
    
    file_ = os.path.join(path_, 'batch_'+fkey_+'_'+str(batch_size), str(batch_id))
    fp = open(file_, 'r')
    src_arr = []
    trg_arr = []
    src_lens = []
    trg_lens = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        dabs = re.split('\s', arr[0])
        dabs = filter(None, dabs) + ['<stop>']
        trg_lens.append(len(dabs))
        
        dabs2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dabs
        ]
        trg_arr.append(dabs2id)
                
        dart = re.split('\s', arr[1])
        dart = filter(None, dart)
        src_lens.append(len(dart))
        dart2id = [
            src_vocab2id[wd] if wd in src_vocab2id
            else src_vocab2id['<unk>']
            for wd in dart
        ]
        src_arr.append(dart2id)
    fp.close()
    
    src_max_lens = max_lens[0]
    trg_max_lens = max_lens[1]
            
    src_arr = [itm[:src_max_lens] for itm in src_arr]
    trg_arr = [itm[:trg_max_lens] for itm in trg_arr]

    src_arr = [
        itm + [src_vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in src_arr
    ]
    trg_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
        for itm in trg_arr
    ]
    trg_output_arr = [
        itm[1:] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
        for itm in trg_arr
    ]
    
    src_var = Variable(torch.LongTensor(src_arr))
    trg_input_var = Variable(torch.LongTensor(trg_input_arr))
    trg_output_var = Variable(torch.LongTensor(trg_output_arr))
    
    return src_var, trg_input_var, trg_output_var
'''
Process the minibatch test
'''
def process_minibatch_test(batch_id, path_, batch_size, vocab2id, src_lens):
    
    file_ = os.path.join(path_, 'batch_test_'+str(batch_size), str(batch_id))
    fp = open(file_, 'r')
    src_arr = []
    src_idx = []
    trg_arr = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        dabs = re.split('\s', arr[0])
        dabs = filter(None, dabs)
        dabs = ' '.join(dabs)
        trg_arr.append(dabs)
        
        dart = re.split('\s', arr[1])
        dart = filter(None, dart)
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart
        ]
        src_idx.append(dart2id)
        src_arr.append(dart)
    fp.close()

    src_idx = [itm[:src_lens] for itm in src_idx]
    src_idx = [
        itm + [vocab2id['<pad>']]*(src_lens-len(itm))
        for itm in src_idx
    ]
    src_var = Variable(torch.LongTensor(src_idx))
    
    src_arr = [itm[:src_lens] for itm in src_arr]
    src_arr = [
        itm + ['<pad>']*(src_lens-len(itm))
        for itm in src_arr
    ]
    
    return src_var, src_arr, trg_arr