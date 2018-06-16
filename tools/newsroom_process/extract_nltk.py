import re
import nltk
from newsroom import jsonl
from multiprocessing import Pool

def process_data(input_):
    article = input_['text']
    summary = input_['summary']
    title = input_['title']
    if article == None or summary == None or title == None:
        return ''
    
    sen_arr = []
    for sen in nltk.sent_tokenize(article):
        sen = nltk.word_tokenize(sen)
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    article = ' '.join(sen_arr)
    sen_arr = []
    for sen in nltk.sent_tokenize(summary):
        sen = nltk.word_tokenize(sen)
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    summary = ' '.join(sen_arr)
    sen_arr = []
    for sen in nltk.sent_tokenize(title):
        sen = nltk.word_tokenize(sen)
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    title = ' '.join(sen_arr)

    sen_arr = [title, summary, article]
    
    return '<sec>'.join(sen_arr)

fout = open('plain_data/test.txt', 'w')
fp = jsonl.open('extract_data/test.data', gzip=True)
cnt = 0
batcher = []
for line in fp:
    cnt += 1
    print(cnt)
    batcher.append(line)
    if len(batcher) == 64:
        pool = Pool(processes=16)
        result = pool.map(process_data, batcher)
        pool.terminate()
        for itm in result:
            if len(itm) > 1:
                fout.write(itm+'\n')
        batcher = []

if len(batcher) > 0:
    pool = Pool(processes=16)
    result = pool.map(process_data, batcher)
    pool.terminate()
    for itm in result:
        if len(itm) > 1:
            fout.write(itm+'\n')
    batcher = []
fp.close()
fout.close()

fout = open('plain_data/dev.txt', 'w')
fp = jsonl.open('extract_data/dev.data', gzip=True)
cnt = 0
batcher = []
for line in fp:
    cnt += 1
    print(cnt)
    batcher.append(line)
    if len(batcher) == 64:
        pool = Pool(processes=16)
        result = pool.map(process_data, batcher)
        pool.terminate()
        for itm in result:
            if len(itm) > 1:
                fout.write(itm+'\n')
        batcher = []

if len(batcher) > 0:
    pool = Pool(processes=16)
    result = pool.map(process_data, batcher)
    pool.terminate()
    for itm in result:
        if len(itm) > 1:
            fout.write(itm+'\n')
    batcher = []
fp.close()
fout.close()

fout = open('plain_data/train.txt', 'w')
fp = jsonl.open('extract_data/train.data', gzip=True)
cnt = 0
batcher = []
for line in fp:
    cnt += 1
    print(cnt)
    batcher.append(line)
    if len(batcher) == 64:
        pool = Pool(processes=16)
        result = pool.map(process_data, batcher)
        pool.terminate()
        for itm in result:
            if len(itm) > 1:
                fout.write(itm+'\n')
        batcher = []

if len(batcher) > 0:
    pool = Pool(processes=16)
    result = pool.map(process_data, batcher)
    pool.terminate()
    for itm in result:
        if len(itm) > 1:
            fout.write(itm+'\n')
    batcher = []
fp.close()
fout.close()