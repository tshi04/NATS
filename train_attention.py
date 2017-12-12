import re
import os
import argparse
import shutil
import glob

import torch
from torch.autograd import Variable

from model import *
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../sum_data/', help='directory that store the data')
parser.add_argument('--file_vocab', default='cnn_vocab.txt', help='vocabulary file')
parser.add_argument('--file_corpus', default='cnn.txt', help='file store documents')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--src_emb_dim', type=int, default=100, help='source embedding dimension')
parser.add_argument('--trg_emb_dim', type=int, default=100, help='target embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=100, help='encoder hidden dimension')
parser.add_argument('--trg_hidden_dim', type=int, default=100, help='decoder hidden dimension')
parser.add_argument('--src_num_layers', type=int, default=2, help='encoder number layers')
parser.add_argument('--trg_num_layers', type=int, default=1, help='decoder number layers')
parser.add_argument('--src_bidirection', type=bool, default=True)
parser.add_argument('--batch_first', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--attn_method', default='bahdanau', help='vanilla | bahdanau | luong_dot | luong_concat | luong_general')
parser.add_argument('--network_', default='gru')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--src_max_lens', type=int, default=256)
parser.add_argument('--trg_max_lens', type=int, default=30)
opt = parser.parse_args()

vocab2id, id2vocab = construct_vocab(opt.data_dir+'/'+opt.file_vocab)
print 'The vocabulary size: {0}'.format(len(vocab2id))

if os.path.exists('batch_folder'):
    n_batch = create_batch_file(
        file_name=opt.data_dir+'/'+opt.file_corpus, 
        batch_size=opt.batch_size
    )
else:
    n_batch = len(glob.glob('batch_folder/batch_*'))
print 'The number of batches: {0}'.format(n_batch)

model = Seq2Seq(
    src_emb_dim=opt.src_emb_dim,
    trg_emb_dim=opt.trg_emb_dim,
    src_hidden_dim=opt.src_hidden_dim,
    trg_hidden_dim=opt.trg_hidden_dim,
    src_vocab_size=len(vocab2id),
    trg_vocab_size=len(vocab2id),
    src_nlayer=opt.src_num_layers,
    trg_nlayer=opt.trg_num_layers,
    batch_first=opt.batch_first,
    src_bidirect=opt.src_bidirection,
    batch_size=opt.batch_size,
    dropout=opt.dropout,
    attn_method=opt.attn_method,
    network_=opt.network_
).cuda()

print model

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
for epoch in range(opt.n_epoch):
    for batch_id in range(n_batch):
        src_var, trg_input_var, trg_output_var = process_minibatch(
            batch_id, vocab2id, max_lens=[opt.src_max_lens, opt.trg_max_lens]
        )
        logits = model(src_var.cuda(), trg_input_var.cuda())
        optimizer.zero_grad()
        
        loss = loss_criterion(
            logits.contiguous().view(-1, len(vocab2id)),
            trg_output_var.view(-1).cuda()
        )
        loss.backward()
        optimizer.step()
        
        losses.append([epoch, batch_id, loss.data.cpu().numpy()[0]])
        if batch_id % 500 == 0:
            loss_np = np.array(losses)
            np.save(out_dir+'/loss', loss_np)
            
            print 'epoch={0} batch={1} loss={2}'.format(
                epoch, batch_id, loss.data.cpu().numpy()[0]
            )
            word_prob = model.decode(logits).data.cpu().numpy().argmax(axis=2)
            sen_pred = [id2vocab[x] for x in word_prob[0]]
            st_idx = len(sen_pred)
            for k, wd in enumerate(sen_pred):
                if wd == '</s>':
                    st_idx = k
                    break
            sen_pred = sen_pred[:st_idx]
            print ' '.join(sen_pred)
            torch.save(
                model,
                open(os.path.join(out_dir, 'seq2seq_'+str(epoch)+'_'+str(batch_id)+'.pt'), 'w')
            )
            
            