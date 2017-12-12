import os
import re
import shutil
import numpy as np


import torch
from torch.autograd import Variable


def construct_vocab(file_, mincount=10):
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
            arr = re.split('<sec>', line[:-1])
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


def process_minibatch(batch_id, vocab2id, max_lens=[512, 64]):
    
    folder = 'batch_folder'
    fkey = 'batch_'
    file_ = folder + '/' + fkey + str(batch_id)
    fp = open(file_, 'r')
    src_arr = []
    trg_arr = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
            
        dabs = re.split('<pg>|<st>', arr[2])
        for j in range(len(dabs)):
            dabs[j] += '.'
        dabs = ''.join(dabs)
        dabs = re.split('\s', dabs)
        dabs = filter(None, dabs)
        dabs = ['<s>'] + dabs + ['</s>']
        dabs2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dabs
        ]
        trg_arr.append(dabs2id)
        
        dart = ''.join(re.split('<pg>|<st>', arr[3]))
        dart = re.split('\s', dart)
        dart = filter(None, dart)
        dart = ['<s>'] + dart + ['</s>']
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart
        ]
        src_arr.append(dart2id)
    fp.close()
    
    src_arr = [itm[:max_lens[0]] for itm in src_arr]
    trg_arr = [itm[:max_lens[1]] for itm in trg_arr]

    src_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+max_lens[0]-len(itm))
        for itm in src_arr
    ]
    trg_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+max_lens[1]-len(itm))
        for itm in trg_arr
    ]
    trg_output_arr = [
        itm[1:] + [vocab2id['<pad>']]*(1+max_lens[1]-len(itm))
        for itm in trg_arr
    ]
    
    src_var = Variable(torch.LongTensor(src_arr))
    trg_input_var = Variable(torch.LongTensor(trg_input_arr))
    trg_output_var = Variable(torch.LongTensor(trg_output_arr))
    
    return src_var, trg_input_var, trg_output_var




