import re
import os
import argparse
import shutil
import glob
import time

import torch
from torch.autograd import Variable

from model import *
from utils import *
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='train', help='train | test | validate | rouge | fastbeam | model2para')
parser.add_argument('--data_dir', default='../sum_data/', help='directory that store the data.')
parser.add_argument('--file_vocab', default='vocab', help='file store training vocabulary.')
parser.add_argument('--file_corpus', default='train.txt', help='file store training documents.')
parser.add_argument('--n_epoch', type=int, default=35, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
parser.add_argument('--src_seq_lens', type=int, default=400, help='length of source documents.')
parser.add_argument('--trg_seq_lens', type=int, default=100, help='length of trage documents.')
parser.add_argument('--src_emb_dim', type=int, default=128, help='source embedding dimension')
parser.add_argument('--trg_emb_dim', type=int, default=128, help='target embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--trg_hidden_dim', type=int, default=256, help='decoder hidden dimension')
parser.add_argument('--attn_hidden_dim', type=int, default=128, help='attn hidden dimension')
parser.add_argument('--src_num_layers', type=int, default=2, help='encoder number layers')
parser.add_argument('--trg_num_layers', type=int, default=1, help='decoder number layers')
parser.add_argument('--vocab_size', type=int, default=50000, help='max number of words in the vocabulary.')
parser.add_argument('--word_mincount', type=int, default=5, 
                    help='min count of the words in the corpus in the vocab')
parser.add_argument('--src_bidirection', type=bool, default=True, help='encoder bidirectional?')
parser.add_argument('--batch_first', type=bool, default=True, help='batch first?')
parser.add_argument('--shared_embedding', type=bool, default=True, help='source / target share embedding?')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--attn_method', default='bahdanau_concat',
                    help='vanilla | bahdanau_dot | bahdanau_concat | luong_dot | luong_concat | luong_general')
parser.add_argument('--coverage', default='simple',
                    help='vanilla | simple | concat | gru | asee')
parser.add_argument('--network_', default='gru', help='gru | lstm')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--debug', type=bool, default=False, help='if true will clean the output after training')
parser.add_argument('--grad_clip', type=float, default=2.0, help='clip the gradient norm.')
parser.add_argument('--clean_batch', type=bool, default=False, help='Do you want to clean the batch folder?')
# used in the test
parser.add_argument('--model_dir', default='seq2seq_results-0', help='directory that store the model.')
parser.add_argument('--model_file', default='seq2seq_34_8000', help='file for model.')
parser.add_argument('--file_test', default='test.txt', help='test data')
parser.add_argument('--beam_size', type=int, default=5, help='beam size.')
opt = parser.parse_args()

vocab2id, id2vocab = construct_vocab(
    file_=opt.data_dir+'/'+opt.file_vocab,
    max_size=opt.vocab_size,
    mincount=opt.word_mincount
)
print 'The vocabulary size: {0}'.format(len(vocab2id))
if opt.task == 'train':
    n_batch = create_batch_file(
        path_=opt.data_dir,
        fkey_='train',
        file_=opt.file_corpus,
        batch_size=opt.batch_size,
        clean=opt.clean_batch
    )
    print 'The number of batches: {0}'.format(n_batch)
elif opt.task == 'test' or opt.task == 'fastbeam':
    test_batch = create_batch_file(
        path_=opt.data_dir,
        fkey_='test',
        file_=opt.file_test,
        batch_size=opt.batch_size,
        clean=False
    )
    print 'The number of batches (test): {0}'.format(test_batch)

if opt.task == 'train' or opt.task == 'fastbeam' or opt.task == 'test':
    model = Seq2Seq(
        src_seq_len=opt.src_seq_lens,
        trg_seq_len=opt.trg_seq_lens,
        src_emb_dim=opt.src_emb_dim,
        trg_emb_dim=opt.trg_emb_dim,
        src_hidden_dim=opt.src_hidden_dim,
        trg_hidden_dim=opt.trg_hidden_dim,
        attn_hidden_dim=opt.attn_hidden_dim,
        src_vocab_size=len(vocab2id),
        trg_vocab_size=len(vocab2id),
        src_nlayer=opt.src_num_layers,
        trg_nlayer=opt.trg_num_layers,
        batch_first=opt.batch_first,
        src_bidirect=opt.src_bidirection,
        dropout=opt.dropout,
        attn_method=opt.attn_method,
        coverage=opt.coverage,
        network_=opt.network_,
        shared_emb=opt.shared_embedding
    ).cuda()
    print model
'''
train
'''
if opt.task == 'train':
    weight_mask = torch.ones(len(vocab2id)).cuda()
    weight_mask[vocab2id['<pad>']] = 0
    loss_criterion = torch.nn.CrossEntropyLoss(weight=weight_mask).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    lead_dir = opt.data_dir+'/seq2seq_results-'
    for k in range(1000000):
        out_dir = lead_dir+str(k)
        if not os.path.exists(out_dir):
            break
    os.mkdir(out_dir)

    losses = []
    start_time = time.time()
    for epoch in range(opt.n_epoch):
        for batch_id in range(n_batch):
            src_var, trg_input_var, trg_output_var = process_minibatch(
                batch_id=batch_id, path_=opt.data_dir, fkey_='train', 
                batch_size=opt.batch_size, vocab2id=vocab2id, 
                max_lens=[opt.src_seq_lens, opt.trg_seq_lens]
            )
            logits, _ = model(src_var.cuda(), trg_input_var.cuda())
            optimizer.zero_grad()
        
            loss = loss_criterion(
                logits.contiguous().view(-1, len(vocab2id)),
                trg_output_var.view(-1).cuda()
            )
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
        
            end_time = time.time()
            losses.append([epoch, batch_id, loss.data.cpu().numpy()[0], (end_time-start_time)/3600.0])
            if batch_id % 2000 == 0:
                loss_np = np.array(losses)
                np.save(out_dir+'/loss', loss_np)
                fmodel = open(os.path.join(out_dir, 'seq2seq_'+str(epoch)+'_'+str(batch_id)+'.pt'), 'w')
                torch.save(model, fmodel)
                fmodel.close()
            if batch_id%100 == 0:
                end_time = time.time()
                word_prob = model.decode(logits).data.cpu().numpy().argmax(axis=2)
                sen_pred = [id2vocab[x] for x in word_prob[0]]
                print 'epoch={0} batch={1} loss={2}, time_escape={3}s={4}h'.format(
                    epoch, batch_id, loss.data.cpu().numpy()[0], 
                    end_time-start_time, (end_time-start_time)/3600.0
                )
                print ' '.join(sen_pred)
            if opt.debug:
                break
        if opt.debug:
            break
        
        loss_np = np.array(losses)
        np.save(out_dir+'/loss', loss_np)
        fmodel = open(os.path.join(out_dir, 'seq2seq_'+str(epoch)+'_'+str(batch_id)+'.pt'), 'w')
        torch.save(model, fmodel)
        fmodel.close()
            
    if opt.debug:
        shutil.rmtree(out_dir)
'''
rouge
'''
if opt.task == 'rouge':
    rouge_path = os.path.join(opt.data_dir, 'rouge')
    if os.path.exists(rouge_path):
        shutil.rmtree(rouge_path)
    os.makedirs(rouge_path)
    sys_smm_path = os.path.join(rouge_path, 'system_summaries')
    mod_smm_path = os.path.join(rouge_path, 'model_summaries')
    os.makedirs(sys_smm_path)
    os.makedirs(mod_smm_path)
    fp = open(os.path.join(opt.data_dir, 'summaries.txt'), 'r')
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
            arr = ['' if wd == '<unk>' else wd for wd in arr]
            arr = filter(None, arr)
            dstr = ' '.join(arr)
            fout.write(dstr+'\n')
        fout.close()
        fout = open(os.path.join(mod_smm_path, 'sum.A.'+str(cnt).zfill(5)+'.txt'), 'w')
        for sen in smm:
            arr = re.split('\s', sen)
            arr = ['' if wd == '<unk>' else wd for wd in arr]
            arr = filter(None, arr)
            dstr = ' '.join(arr)
            fout.write(dstr+'\n')
        fout.close()
        cnt += 1
    fp.close()
'''
model2para
'''
if opt.task == 'model2para':
    models_ = glob.glob(os.path.join(opt.data_dir, opt.model_dir, '*.pt'))
    for file_ in models_:
        out_file = re.split('.pt', file_)[0]
        model = torch.load(file_)
        torch.save(model.state_dict(), out_file+'.model')
        print '{0}->{1}'.format(file_, out_file)
'''
test
'''
if opt.task == 'test':
    model.load_state_dict(torch.load(
        os.path.join(opt.data_dir, opt.model_dir, opt.model_file+'.model')))

    start_time = time.time()
    fout = open(os.path.join(opt.data_dir, 'summaries.txt'), 'w')
    for batch_id in range(test_batch):
        src_var, trg_input_var, trg_output_var = process_minibatch(
            batch_id=batch_id, path_=opt.data_dir, fkey_='test', 
            batch_size=opt.batch_size, vocab2id=vocab2id, 
            max_lens=[opt.src_seq_lens, opt.trg_seq_lens]
        )
        beam_seq, beam_prb = batch_beam_search(
            model=model, 
            src_text=src_var, 
            vocab2id=vocab2id, 
            beam_size=opt.beam_size, 
            max_len=opt.trg_seq_lens
        )
        trg_seq = trg_output_var.data.numpy()
        for b in range(trg_seq.shape[0]):
            arr = []
            gen_text = beam_seq.data.cpu().numpy()[b,0]
            gen_text = [id2vocab[wd] for wd in gen_text]
            arr.append(' '.join(gen_text))
            trg_text = [id2vocab[wd] for wd in trg_seq[b]]
            arr.append(' '.join(trg_text))
            fout.write('<sec>'.join(arr)+'\n')

        end_time = time.time()
        print(batch_id, end_time-start_time)
    
    fout.close()
'''
fastbeam
'''
if opt.task == 'fastbeam':
    model.load_state_dict(torch.load(
        os.path.join(opt.data_dir, opt.model_dir, opt.model_file+'.model')))
    
    start_time = time.time()
    fout = open(os.path.join(opt.data_dir, 'summaries.txt'), 'w')
    for batch_id in range(test_batch):
        src_var, trg_input_var, trg_output_var = process_minibatch(
            batch_id=batch_id, path_=opt.data_dir, fkey_='test', 
            batch_size=opt.batch_size, vocab2id=vocab2id, 
            max_lens=[opt.src_seq_lens, opt.trg_seq_lens]
        )
        beam_seq, beam_prb = fast_beam_search(
            model=model, 
            src_text=src_var, 
            vocab2id=vocab2id, 
            beam_size=opt.beam_size, 
            max_len=opt.trg_seq_lens
        )
        trg_seq = trg_output_var.data.numpy()
        for b in range(trg_seq.shape[0]):
            arr = []
            gen_text = beam_seq.data.cpu().numpy()[b,0]
            gen_text = [id2vocab[wd] for wd in gen_text]
            arr.append(' '.join(gen_text))
            trg_text = [id2vocab[wd] for wd in trg_seq[b]]
            arr.append(' '.join(trg_text))
            fout.write('<sec>'.join(arr)+'\n')

        end_time = time.time()
        print(batch_id, end_time-start_time)
    
    fout.close()
    
'''
validate
'''
if opt.task == 'validate':
    print 'good'
