import re
import numpy as np

data_dir = '../data4dl/seq2seq-st/data/'
word_vec = np.load(data_dir+'word_vec.npy')

article_ = []
fp = open(data_dir+'article_idx.txt', 'r')
for line in fp:
    arr = re.split('\s', line[:-1])
    article_.append(arr)
fp.close()

title_ = []
fp = open(data_dir+'title_idx.txt', 'r')
for line in fp:
    arr = re.split('\s', line[:-1])
    title_.append(arr)
fp.close()

model = seq2seq(
    src_emb_dim=100,
    trg_emb_dim=100,
    src_hidden_dim=25,
    trg_hidden_dim=50,
    src_vocab_size=141570,
    trg_vocab_size=141570,
    src_pad_token=0,
    trg_pad_token=0,
    src_nlayer=2,
    trg_nlayer=1,
    src_bidirect=True,
    batch_size=128,
    dropout=0.
).cuda()

loss_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for k in range(100):
    
    
    decoder_logit = model(input_lines_src, input_lines_trg)
    optimizer.zero_grad()