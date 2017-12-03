import argparse
import shutil
import re
import os

import torch
from torch.autograd import Variable

from seq2seq import *
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='seq2seq', help='seq2seq | seqGAN')
stc = parser.parse_args()

data_dir = '../sum_data/'
file_vocab = 'cnn_vocab.txt'
file_corpus = 'cnn.txt'
n_epoch = 20
batch_size = 64

vocab2id, id2vocab = construct_vocab(data_dir+'/'+file_vocab)
print 'The vocabulary size: {0}'.format(len(vocab2id))

n_batch = create_batch_file(file_name='../sum_data/cnn.txt', batch_size=batch_size)
print 'The number of batches: {0}'.format(n_batch)

model = seq2seq(
    src_emb_dim=100,
    trg_emb_dim=100,
    src_hidden_dim=25,
    trg_hidden_dim=50,
    src_vocab_size=len(vocab2id),
    trg_vocab_size=len(vocab2id),
    src_pad_token=0,
    trg_pad_token=0,
    src_nlayer=2,
    trg_nlayer=1,
    src_bidirect=True,
    batch_size=batch_size,
    dropout=0.0
).cuda()

weight_mask = torch.ones(len(vocab2id)).cuda()
weight_mask[vocab2id['<pad>']] = 0
loss_criterion = torch.nn.CrossEntropyLoss(weight=weight_mask).cuda()

optimizer = torch.optim.Adam(model.parameters())

out_dir = 'results'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
losses = []
for epoch in range(n_epoch):
    for batch_id in range(n_batch):
        src_var, trg_input_var, trg_output_var = process_minibatch(
            batch_id, vocab2id, max_lens=[512, 64]
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
                model.state_dict(),
                open(os.path.join(out_dir, 'lstm2lstm_'+str(epoch)+'_'+str(batch_id)+'.model'), 'w')
            )
                       
shutil.rmtree('batch_folder')
