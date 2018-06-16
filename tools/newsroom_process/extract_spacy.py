import re
import spacy
import time
nlp = spacy.load('en', disable=['tagger', 'ner'])
from newsroom import jsonl
from multiprocessing import Pool

def process_data(input_):
    article = input_['text']
    summary = input_['summary']
    title = input_['title']
    if article == None or summary == None or title == None:
        return ''
    article = nlp(article)
    summary = nlp(summary)
    title = nlp(title)
    sen_arr = []
    for sen in article.sents:
        sen = [k.text for k in sen if '\n' not in k.text]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    article = ' '.join(sen_arr)
    sen_arr = []
    for sen in summary.sents:
        sen = [k.text for k in sen if '\n' not in k.text]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    summary = ' '.join(sen_arr)
    sen_arr = []
    for sen in title.sents:
        sen = [k.text for k in sen if '\n' not in k.text]
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
start = time.time()
end = time.time()
for line in fp:
    cnt += 1
    print(cnt, end-start)
    batcher.append(line)
    if len(batcher) == 64:
        pool = Pool(processes=16)
        result = pool.map(process_data, batcher)
        pool.terminate()
        for itm in result:
            if len(itm) > 1:
                fout.write(itm+'\n')
        batcher = []
        end = time.time()

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
    print(cnt, end-start)
    batcher.append(line)
    if len(batcher) == 64:
        pool = Pool(processes=16)
        result = pool.map(process_data, batcher)
        pool.terminate()
        for itm in result:
            if len(itm) > 1:
                fout.write(itm+'\n')
        batcher = []
        end = time.time()

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
    print(cnt, end-start)
    batcher.append(line)
    if len(batcher) == 64:
        pool = Pool(processes=16)
        result = pool.map(process_data, batcher)
        pool.terminate()
        for itm in result:
            if len(itm) > 1:
                fout.write(itm+'\n')
        batcher = []
        end = time.time()

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