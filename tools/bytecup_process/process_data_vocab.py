import re
import os
import sys
import glob
import json

def show_progress(a, b, time=""):
    cc = int(round(100.0*float(a)/float(b)))
    dstr = '[' + '>'*cc + ' '*(100-cc) + ']'
    sys.stdout.write(dstr + str(cc) + '%' + time +'\r')
    sys.stdout.flush()

print('read split ...')
train_id = {}
val_id = {}
test_id = {}
fp = open('data_split/train_id.txt', 'r')
for line in fp:
    train_id[line[:-1]] = {}
fp.close()
fp = open('data_split/val_id.txt', 'r')
for line in fp:
    val_id[line[:-1]] = {}
fp.close()
fp = open('data_split/test_id.txt', 'r')
for line in fp:
    test_id[line[:-1]] = {}
fp.close()
total_ = len(train_id) + len(val_id) + len(val_id)

print('combine data')
files_ = glob.glob('new*.txt')
fout1 = open('train.txt', 'w')
fout2 = open('val.txt', 'w')
fout3 = open('test.txt', 'w')
cnt = 0
for fl in files_:
    fp = open(fl, 'r')
    for line in fp:
        cnt += 1
        show_progress(cnt, total_)
        arr = json.loads(line)
        if str(arr['id']) in train_id:
            title = arr['title']
            artile = re.split('\s|<s>|</s>', arr['content'])
            artile = ' '.join(list(filter(None, artile)))
            fout1.write('<sec>'.join([title, artile])+'\n')
        if str(arr['id']) in val_id:
            title = arr['title']
            artile = re.split('\s|<s>|</s>', arr['content'])
            artile = ' '.join(list(filter(None, artile)))
            fout2.write('<sec>'.join([title, artile])+'\n')
        if str(arr['id']) in test_id:
            title = arr['title']
            artile = re.split('\s|<s>|</s>', arr['content'])
            artile = ' '.join(list(filter(None, artile)))
            fout3.write('<sec>'.join([title, artile])+'\n')
    fp.close()
fout1.close()
fout2.close()
fout3.close()

print()
print('-- create vocab title')
vocab = {}
cnt = 0
fp = open('train.txt', 'r')
for line in fp:
    arr = re.split('<s>|</s>|<sec>|\s', line[:-1].lower())
    arr = filter(None, arr)
    for wd in arr:
        try:
            vocab[wd] += 1
        except:
            vocab[wd] = 1
    cnt += 1
    show_progress(cnt, len(train_id))
fp.close()
show_progress(len(train_id), len(train_id))
print()
print(len(vocab))
print('-- write vocab')
vocab_arr = [[wd, vocab[wd]] for wd in vocab]
vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
fout = open('vocab', 'w')
for itm in vocab_arr:
    itm[1] = str(itm[1])
    fout.write(' '.join(itm)+'\n')
fout.close()
