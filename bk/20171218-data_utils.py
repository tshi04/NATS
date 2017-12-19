import os
import re
import shutil
import numpy as np

import torch
from torch.autograd import Variable

def construct_vocab(file_, mincount=5):
    vocab2id = {
        '<s>': 0,
        '</s>': 1,
        '<pad>': 2,
        '<unk>': 3
    }
    
    id2vocab = {
        0: '<s>',
        1: '</s>',
        2: '<pad>',
        3: '<unk>'
    }
    cnt = 4
    with open(file_, 'r') as fp:
        for line in fp:
            try:
                arr = re.split('<sec>', line[:-1])
                arr[1]
            except:
                arr = re.split('\s', line[:-1])
            if arr[0] == ' ':
                continue
            if int(arr[1]) >= mincount:
                vocab2id[arr[0]] = cnt
                id2vocab[cnt] = arr[0]
                cnt += 1
    
    return vocab2id, id2vocab

def create_batch_file(file_name, batch_size):
    folder = 'batch_folder'
    fkey = 'batch_'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    
    fp = open(file_name, 'r')
    cnt = 0
    for line in fp:
        try:
            arr.append(line)
        except:
            arr = []
        if len(arr) == batch_size:
            fout = open(folder+'/'+fkey+str(cnt), 'w')
            for itm in arr:
                fout.write(itm)
            fout.close()
            arr = []
            cnt += 1
    fp.close()
    
    return cnt

def process_minibatch(batch_id, vocab2id, max_lens=[100, 20]):
    
    folder = 'batch_folder'
    fkey = 'batch_'
    file_ = folder + '/' + fkey + str(batch_id)
    fp = open(file_, 'r')
    src_arr = []
    trg_arr = []
    src_lens = []
    trg_lens = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])

        dabs = re.split('\s', arr[0])
        dabs = filter(None, dabs)
        dabs = ['<s>'] + dabs + ['</s>']
        trg_lens.append(len(dabs))
        dabs2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dabs
        ]
        trg_arr.append(dabs2id)
        
        dart = re.split('\s', arr[1])
        dart = filter(None, dart)
        dart = ['<s>'] + dart + ['</s>']
        src_lens.append(len(dart))
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart
        ]
        src_arr.append(dart2id)
    fp.close()
    
    src_max_lens = max(src_lens) 
    if max_lens[0] < max(src_lens):
        src_max_lens = max_lens[0]
    trg_max_lens = max(trg_lens)
    if max_lens[1] < max(trg_lens):
        trg_max_lens = max(trg_lens)
    
    src_arr = [itm[:max_lens[0]] for itm in src_arr]
    trg_arr = [itm[:max_lens[1]] for itm in trg_arr]

    src_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+src_max_lens-len(itm))
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

