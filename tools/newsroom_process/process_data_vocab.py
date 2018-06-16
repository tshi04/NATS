import re
import os
import sys
import shutil

def show_progress(a, b):
    cc = int(round(100.0*float(a)/float(b)))
    dstr = '[' + '>'*cc + ' '*(100-cc) + ']'
    sys.stdout.write(dstr + str(cc) + '%' +'\r')
    sys.stdout.flush()

plain_dir = 'plain_data'
tmp_dir = 'tmp_data'
title_dir = 'title_data'
sum_dir = 'sum_data'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.mkdir(tmp_dir)
if os.path.exists(title_dir):
    shutil.rmtree(title_dir)
os.mkdir(title_dir)
if os.path.exists(sum_dir):
    shutil.rmtree(sum_dir)
os.mkdir(sum_dir)

print('-- remove special characters')
cnt = 0
fout = open(os.path.join(tmp_dir, 'train.txt'), 'w')
fp = open(os.path.join(plain_dir, 'train.txt'), 'rb')
for line in fp:
    line = line.decode('utf-8').encode('ascii', 'ignore').decode('ascii')
    fout.write(line)
    cnt += 1
    show_progress(cnt, 1200000)
fp.close()
fout.close()
show_progress(1200000, 1200000)

fout = open(os.path.join(tmp_dir, 'dev.txt'), 'w')
fp = open(os.path.join(plain_dir, 'dev.txt'), 'rb')
for line in fp:
    line = line.decode('utf-8').encode('ascii', 'ignore').decode('ascii')
    fout.write(line)
    cnt += 1
    show_progress(cnt, 1200000)
fp.close()
fout.close()
show_progress(1200000, 1200000)

fout = open(os.path.join(tmp_dir, 'test.txt'), 'w')
fp = open(os.path.join(plain_dir, 'test.txt'), 'rb')
for line in fp:
    line = line.decode('utf-8').encode('ascii', 'ignore').decode('ascii')
    fout.write(line)
    cnt += 1
    show_progress(cnt, 1200000)
fp.close()
fout.close()
show_progress(1200000, 1200000)

print()
print('-- create dataset')
fout0 = open(os.path.join(title_dir, 'train.txt'), 'w')
fout1 = open(os.path.join(sum_dir, 'train.txt'), 'w')
fp = open(os.path.join(tmp_dir, 'train.txt'), 'r')
cnt = 0
for line in fp:
    try:
        arr = re.split('<sec>', line[:-1])
        title = arr[0]
        summary = arr[1]
        article = re.split('<s>|</s>|\s', arr[2])
    except:
        continue
    article = list(filter(None, article))
    article = ' '.join(article)
    fout0.write('<sec>'.join([title.lower(), article.lower()])+'\n')
    fout1.write('<sec>'.join([summary.lower(), article.lower()])+'\n')
    cnt += 1
    show_progress(cnt, 1200000)
fp.close()
fout0.close()
fout1.close()
show_progress(1200000, 1200000)

fout0 = open(os.path.join(title_dir, 'val.txt'), 'w')
fout1 = open(os.path.join(sum_dir, 'val.txt'), 'w')
fp = open(os.path.join(tmp_dir, 'dev.txt'), 'r')
for line in fp:
    try:
        arr = re.split('<sec>', line[:-1])
        title = arr[0]
        summary = arr[1]
        article = re.split('<s>|</s>|\s', arr[2])
    except:
        continue
    article = list(filter(None, article))
    article = ' '.join(article)
    fout0.write('<sec>'.join([title.lower(), article.lower()])+'\n')
    fout1.write('<sec>'.join([summary.lower(), article.lower()])+'\n')
    cnt += 1
    show_progress(cnt, 1200000)
fp.close()
fout0.close()
fout1.close()
show_progress(1200000, 1200000)

fout0 = open(os.path.join(title_dir, 'test.txt'), 'w')
fout1 = open(os.path.join(sum_dir, 'test.txt'), 'w')
fp = open(os.path.join(tmp_dir, 'test.txt'), 'r')
for line in fp:
    try:
        arr = re.split('<sec>', line[:-1])
        title = arr[0]
        summary = arr[1]
        article = re.split('<s>|</s>|\s', arr[2])
    except:
        continue
    article = list(filter(None, article))
    article = ' '.join(article)
    fout0.write('<sec>'.join([title.lower(), article.lower()])+'\n')
    fout1.write('<sec>'.join([summary.lower(), article.lower()])+'\n')
    cnt += 1
    show_progress(cnt, 1200000)
fp.close()
fout0.close()
fout1.close()
show_progress(1200000, 1200000)

print()
print('-- create vocab title')
vocab = {}
cnt = 0
fp = open(os.path.join(title_dir, 'train.txt'), 'r')
for line in fp:
    arr = re.split('<s>|</s>|<sec>|\s', line[:-1].lower())
    arr = filter(None, arr)
    for wd in arr:
        try:
            vocab[wd] += 1
        except:
            vocab[wd] = 1
    cnt += 1
    show_progress(cnt, 1000000)
fp.close()
show_progress(1000000, 1000000)
print()
print(len(vocab))
print('-- write vocab')
vocab_arr = [[wd, vocab[wd]] for wd in vocab]
vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
fout = open(os.path.join(title_dir, 'vocab'), 'w')
for itm in vocab_arr:
    itm[1] = str(itm[1])
    fout.write(' '.join(itm)+'\n')
fout.close()

print('-- create vocab summary')
vocab = {}
cnt = 0
fp = open(os.path.join(sum_dir, 'train.txt'), 'r')
for line in fp:
    arr = re.split('<s>|</s>|<sec>|\s', line[:-1].lower())
    arr = filter(None, arr)
    for wd in arr:
        try:
            vocab[wd] += 1
        except:
            vocab[wd] = 1
    cnt += 1
    show_progress(cnt, 1000000)
fp.close()
show_progress(1000000, 1000000)
print()
print(len(vocab))
print('-- write vocab')
vocab_arr = [[wd, vocab[wd]] for wd in vocab]
vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
fout = open(os.path.join(sum_dir, 'vocab'), 'w')
for itm in vocab_arr:
    itm[1] = str(itm[1])
    fout.write(' '.join(itm)+'\n')
fout.close()

