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

def process_minibatch(batch_id, vocab2id, max_len=[10, 100]):
    
    folder = 'batch_folder'
    fkey = 'batch_'
    file_ = folder + '/' + fkey + str(batch_id)
    fp = open(file_, 'r')
    var_art = []
    var_abs = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
            
        dabs = re.split('<pg>', arr[2])
        for j in range(len(dabs)):
            dabs[j] += '.'
        dabs = ''.join(dabs)
        dabs = re.split('\s', dabs)
        dabs = filter(None, dabs)
        dabs = ['<s>'] + dabs + ['</s>']
        dabs_arr = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dabs
        ]
        dabs_arr = dabs_arr[:max_len[0]]
        dabs_arr += [vocab2id['<pad>']]*(max_len[0]-len(dabs_arr))
        var_abs.append(dabs_arr)

        dart = ''.join(re.split('<pg>|<st>', arr[3]))
        dart = re.split('\s', dart)
        dart = filter(None, dart)
        dart = ['<s>'] + dart + ['</s>']
        dart_arr = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart
        ]
        dart_arr = dart_arr[:max_len[1]]
        dart_arr += [vocab2id['<pad>']]*(max_len[1]-len(dart_arr))
        var_art.append(dart_arr)
    fp.close()
    
    var_abs = Variable(torch.LongTensor(var_abs))
    var_art = Variable(torch.LongTensor(var_art))
    
    return var_abs, var_art



