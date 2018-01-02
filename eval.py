import re
import os
import argparse
import shutil
import glob
import time

data_dir = '../sum_data/'
rouge_dir = 'rouge'
file_gen = 'summaries.txt'

rouge_path = os.path.join(data_dir, rouge_dir)
sys_smm_path = os.path.join(rouge_path, 'system_summaries')
mod_smm_path = os.path.join(rouge_path, 'model_summaries')
shutil.rmtree(rouge_path)
os.makedirs(rouge_path)
os.makedirs(sys_smm_path)
os.makedirs(mod_smm_path)
fp = open(os.path.join(data_dir, file_gen), 'r')
cnt = 1
for line in fp:
    arr = re.split('<sec>', line[:-1])
    smm = re.split('<pad>|<s>|</s>', arr[0])
    rmm = re.split('<pad>|<s>|</s>', arr[1])
    rmm = filter(None, rmm)
    smm = filter(None, smm)
    rmm = [' '.join(filter(None, re.split('\s', sen))) for sen in rmm]
    smm = [' '.join(filter(None, re.split('\s', sen))) for sen in smm]
    rmm = filter(None, rmm)
    smm = filter(None, smm)[:3]
    fout = open(os.path.join(sys_smm_path, 'sum.'+str(cnt).zfill(5)+'.txt'), 'w')
    for sen in rmm:
        arr = re.split('\s', sen)
        arr = ['[unk]' if wd == '<unk>' else wd for wd in arr]
        dstr = ' '.join(arr)
        fout.write(dstr+'\n')
    fout.close()
    fout = open(os.path.join(mod_smm_path, 'sum.A.'+str(cnt).zfill(5)+'.txt'), 'w')
    for sen in smm:
        arr = re.split('\s', sen)
        arr = ['[unk]' if wd == '<unk>' else wd for wd in arr]
        dstr = ' '.join(arr)
        fout.write(dstr+'\n')
    fout.close()
    cnt += 1
fp.close()

from pyrouge import Rouge155

r = Rouge155()
r.system_dir = '/home/tian/rouge/system_summaries/'
r.model_dir = '/home/tian/rouge/model_summaries/'
r.system_filename_pattern = 'sum.(\d+).txt'
r.model_filename_pattern = 'sum.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
