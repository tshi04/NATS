import re
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
import time
import json
import argparse
from multiprocessing import Pool

def process_data(input_):
    article = input_['content']
    title = input_['title']
    if article == None or title == None:
        return ''
    article = article[:99999]
    article = nlp.annotate(article, properties={'annotators': 'ssplit', 'outputFormat': 'json'})
    title = nlp.annotate(title, properties={'annotators': 'ssplit', 'outputFormat': 'json'})
    try:
        article['sentences']
        title['sentences']
    except:
        return ''
    
    sen_arr = []
    for sen in article['sentences']:
        sen = [s['originalText'] for s in sen['tokens'] if not '\n' in s['originalText']]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    article = ' '.join(sen_arr)
    
    sen_arr = []
    for sen in title['sentences']:
        sen = [s['originalText'] for s in sen['tokens'] if not '\n' in s['originalText']]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    title = ' '.join(sen_arr)
    
    input_['content'] = article
    input_['title'] = title
    
    return input_

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='bytecup.corpus.train.1.txt', help='input')
parser.add_argument('--output', default='new1.txt', help='output')
args = parser.parse_args()

fout = open(args.output, 'w')
fp = open(args.input, 'r')
cnt = 0
batcher = []
start = time.time()
end = time.time()
for line in fp:
    cnt += 1
    print(cnt, end-start)
    batcher.append(json.loads(line))
    if len(batcher) == 64:
        pool = Pool(processes=16)
        result = pool.map(process_data, batcher)
        pool.terminate()
        for itm in result:
            if len(itm) > 1:
                json.dump(itm, fout)
                fout.write('\n')
        batcher = []
        end = time.time()

if len(batcher) > 0:
    pool = Pool(processes=16)
    result = pool.map(process_data, batcher)
    pool.terminate()
    for itm in result:
        if len(itm) > 1:
            json.dump(itm, fout)
            fout.write('\n')
    batcher = []
fp.close()
fout.close()
