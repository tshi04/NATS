import numpy as np
import torch
from torch.autograd import Variable

class seq2seq(torch.nn.Module):
    '''
    LSTM encoder
    LSTM decoder
    '''
    def __init__(
        self,
        src_emb_dim=100,
        trg_emb_dim=100,
        src_hidden_dim=50,
        trg_hidden_dim=50,
        src_vocab_size=999,
        trg_vocab_size=999,
        src_pad_token=0,
        trg_pad_token=0,
        src_nlayer=2,
        trg_nlayer=1,
        src_bidirect=True,
        batch_size=128,
        dropout=0.0
    ):
        super(seq2seq, self).__init__()
        
        self.src_bidirect = src_bidirect
        self.trg_vocab_size = trg_vocab_size

        self.n_directions = 1
        self.src_hidden_dim = src_hidden_dim
        if src_bidirect:
            self.n_directions = 2
            self.src_hidden_dim = src_hidden_dim // 2
        
        self.src_embedding = torch.nn.Embedding(
            src_vocab_size,
            src_emb_dim,
            padding_idx=0
        ).cuda()
        
        self.trg_embedding = torch.nn.Embedding(
            trg_vocab_size,
            trg_emb_dim,
            padding_idx=0
        ).cuda()
        
        self.encoder = torch.nn.LSTM(
            input_size=src_emb_dim,
            hidden_size=self.src_hidden_dim,
            num_layers=src_nlayer,
            bidirectional=src_bidirect,
            batch_first=True,
            dropout=dropout
        ).cuda()
        
        self.decoder = torch.nn.LSTM(
            input_size=trg_emb_dim,
            hidden_size=trg_hidden_dim,
            num_layers=trg_nlayer,
            batch_first=True,
            dropout=dropout
        ).cuda()
        
        self.src2trg = torch.nn.Linear(
            self.src_hidden_dim*self.n_directions,
            trg_hidden_dim
        ).cuda()
        
        self.trg2vocab = torch.nn.Linear(
            trg_hidden_dim,
            trg_vocab_size
        ).cuda()
        
        # init weights
        torch.nn.init.normal(self.src_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal(self.trg_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(self.src2trg.bias, 0.0)
        torch.nn.init.constant(self.trg2vocab.bias, 0.0)
        

    def forward(self, input_src, input_trg):
        # init state
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)
        
        batch_size = input_src.size(1)
        if self.encoder.batch_first:
            batch_size = input_src.size(0)
            
        src_h_0 = Variable(torch.zeros(
            self.encoder.num_layers*self.n_directions,
            batch_size,
            self.src_hidden_dim
        )).cuda()
        
        src_c_0 = Variable(torch.zeros(
            self.encoder.num_layers*self.n_directions,
            batch_size,
            self.src_hidden_dim
        )).cuda()
                
        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb,
            (src_h_0, src_c_0)
        )
        
        if self.src_bidirect:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
            
        trg_init_state = self.src2trg(h_t)
        trg_init_state = torch.nn.Tanh()(trg_init_state)

        trg_h_0 = trg_init_state.view(
            self.decoder.num_layers,
            trg_init_state.size(0),
            trg_init_state.size(1)
        )
        trg_c_0 = c_t.view(
            self.decoder.num_layers,
            c_t.size(0),
            c_t.size(1)
        )
        
        trg_h, (_, _) = self.decoder(
            trg_emb,
            (trg_h_0, trg_c_0)
        )
        
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0)*trg_h.size(1),
            trg_h.size(2)
        )
                
        decoder_output = self.trg2vocab(trg_h_reshape)
        decoder_output = decoder_output.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_output.size(1)
        )
        
        return decoder_output
    

    def decode(self, logits):
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = torch.nn.functional.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs
