import re
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
import time
import json
from newsroom import jsonl
from multiprocessing import Pool

def process_data(input_):
    article = input_['text']
    summary = input_['summary']
    title = input_['title']
    if article == None or summary == None or title == None:
        return ''
    article = article[:99999]
    article = nlp.annotate(article, properties={'annotators': 'ssplit', 'outputFormat': 'json'})
    summary = nlp.annotate(summary, properties={'annotators': 'ssplit', 'outputFormat': 'json'})
    title = nlp.annotate(title, properties={'annotators': 'ssplit', 'outputFormat': 'json'})
    sen_arr = []
    try:
        article['sentences']
        summary['sentences']
        title['sentences']
    except:
        return ''
        
    for sen in article['sentences']:
        sen = [s['originalText'] for s in sen['tokens'] if not '\n' in s['originalText']]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    article = ' '.join(sen_arr)
    
    sen_arr = []
    for sen in summary['sentences']:
        sen = [s['originalText'] for s in sen['tokens'] if not '\n' in s['originalText']]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    summary = ' '.join(sen_arr)
    
    sen_arr = []
    for sen in title['sentences']:
        sen = [s['originalText'] for s in sen['tokens'] if not '\n' in s['originalText']]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    title = ' '.join(sen_arr)
    
    sen_arr = [title, summary, article]
    
    return '<sec>'.join(sen_arr)

fout = open('cornlp_data/test.txt', 'w')
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

fout = open('cornlp_data/dev.txt', 'w')
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

fout = open('cornlp_data/train.txt', 'w')
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
