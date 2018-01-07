'''
Copyright 2017 @ Tian Shi
@author Tian Shi
Please contact tshi at vt dot edu.

This framework is not flexiable in the length of encoder and decoder,
since we need to use the coverage mechanism.
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable
'''
Bahdanau, D., Cho, K., & Bengio, Y. (2014). 
Neural machine translation by jointly learning to align and translate. 
arXiv preprint arXiv:1409.0473.
Coverage:
See, A., Liu, P. J., & Manning, C. D. (2017). 
Get To The Point: Summarization with Pointer-Generator Networks. 
arXiv preprint arXiv:1704.04368.
'''
class AttentionBahdanau(torch.nn.Module):

    def __init__(
        self,
        src_seq_len,
        trg_seq_len,
        attn_hidden_size,
        hidden_size,
        attn_method,
        coverage,
    ):
        super(AttentionBahdanau, self).__init__()   
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.attn_method = attn_method.lower()
        self.coverage = coverage
        self.hidden_size = hidden_size
        self.attn_hidden_size = attn_hidden_size

        if self.attn_method == 'bahdanau_concat':
            self.attn_en_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=True
            ).cuda()
            self.attn_de_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=True
            ).cuda()
            if self.coverage == 'asee':
                self.attn_cv_in = torch.nn.Linear(1, self.hidden_size, bias=True).cuda()
            self.attn_out = torch.nn.Linear(self.hidden_size, 1, bias=False).cuda()
            
        if self.coverage == 'concat':
            self.cover_in = torch.nn.Linear(
                self.src_seq_len*2,
                self.src_seq_len
            ).cuda()
        elif self.coverage == 'gru':
            self.gru_ = torch.nn.GRUCell(
                self.src_seq_len, 
                self.attn_hidden_size
            ).cuda()
            self.gru_out = torch.nn.Linear(
                self.attn_hidden_size,
                self.src_seq_len
            ).cuda()
            
    def forward(self, last_dehy, enhy, past_attn, hidden_attn):

        if self.attn_method == 'bahdanau_dot':
            attn = torch.bmm(enhy, last_dehy.unsqueeze(2)).squeeze(2)
        if self.attn_method[:15] == 'bahdanau_concat':
            attn_agg = self.attn_en_in(enhy) + self.attn_de_in(last_dehy.unsqueeze(1))
            if self.coverage == 'asee':
                attn_agg = attn_agg + self.attn_cv_in(past_attn.unsqueeze(2))
            attn = self.attn_out(F.tanh(attn_agg)).squeeze(2)
            
        if self.coverage == 'simple':
            attn = attn - past_attn
        elif self.coverage == 'concat':
            attn = self.cover_in(torch.cat((attn, past_attn), 1))
        elif self.coverage == 'gru':
            hidden_attn = self.gru_(attn, hidden_attn)
            attn = self.gru_out(hidden_attn)

        attn = F.softmax(attn, dim=1)
        attn2 = attn.view(attn.size(0), 1, attn.size(1))
        h_attn = torch.bmm(attn2, enhy).squeeze(1)
        #h_attn = F.tanh(h_attn)

        return h_attn, attn, hidden_attn
'''
Luong, M. T., Pham, H., & Manning, C. D. (2015). 
Effective approaches to attention-based neural machine translation. 
arXiv preprint arXiv:1508.04025.
'''
class AttentionLuong(torch.nn.Module):
    
    def __init__(
        self,
        src_seq_len,
        trg_seq_len,
        attn_hidden_size,
        hidden_size,
        attn_method,
        coverage,
    ):
        super(AttentionLuong, self).__init__()
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.method = attn_method.lower()
        self.hidden_size = hidden_size
        self.coverage = coverage
        self.attn_hidden_size = attn_hidden_size
        
        if self.method == 'luong_concat':
            self.attn_en_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=False
            ).cuda()
            self.attn_de_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=False
            ).cuda()
            if self.coverage == 'asee':
                self.attn_cv_in = torch.nn.Linear(1, self.hidden_size, bias=True).cuda()
            self.attn_warp_in = torch.nn.Linear(self.hidden_size, 1, bias=False).cuda()
        if self.method == 'luong_general':
            self.attn_in = torch.nn.Linear(
                self.hidden_size, 
                self.hidden_size,
                bias=True
            ).cuda()
                
        self.attn_out = torch.nn.Linear(
            self.hidden_size*2,
            self.hidden_size,
            bias=False
        ).cuda()
        
        if self.coverage == 'concat':
            self.cover_in = torch.nn.Linear(
                self.src_seq_len*2,
                self.src_seq_len
            ).cuda()
        elif self.coverage == 'gru':
            self.gru_ = torch.nn.GRUCell(
                self.src_seq_len, 
                self.attn_hidden_size
            ).cuda()
            self.gru_out = torch.nn.Linear(
                self.attn_hidden_size,
                self.src_seq_len
            ).cuda()            
        
    def forward(self, dehy, enhy, past_attn, hidden_attn):
        
        if self.method == 'luong_concat':
            attn_agg = self.attn_en_in(enhy) + self.attn_de_in(dehy.unsqueeze(1))
            if self.coverage == 'asee':
                attn_agg = attn_agg + self.attn_cv_in(past_attn.unsqueeze(2))
            attn_agg = F.tanh(attn_agg)
            attn = self.attn_warp_in(attn_agg).squeeze(2)
        else:
            if self.method == 'luong_general':
                enhy_new = self.attn_in(enhy)
                attn = torch.bmm(enhy_new, dehy.unsqueeze(2)).squeeze(2)
            else:
                attn = torch.bmm(enhy, dehy.unsqueeze(2)).squeeze(2)
            
        if self.coverage == 'simple':
            attn -= past_attn
        elif self.coverage == 'concat':
            attn = self.cover_in(torch.cat((attn, past_attn), 1))
        elif self.coverage == 'gru':
            hidden_attn = self.gru_(attn, hidden_attn)
            attn = self.gru_out(hidden_attn)
        
        attn = F.softmax(attn, dim=1)
        attn2 = attn.view(attn.size(0), 1, attn.size(1))

        attn_enhy = torch.bmm(attn2, enhy).squeeze(1)
        
        h_attn = self.attn_out(torch.cat((attn_enhy, dehy), 1))
        h_attn = F.tanh(h_attn)

        return h_attn, attn, hidden_attn
'''
LSTM decoder
'''    
class LSTMDecoder(torch.nn.Module):
    def __init__(
        self,
        src_seq_len,
        trg_seq_len,
        attn_hidden_size,
        input_size, # embedding size
        hidden_size, # h size
        num_layers,
        attn_method,
        coverage,
        attn_as_input,
        batch_first
    ):
        super(LSTMDecoder, self).__init__()
        # parameters
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.attn_hidden_size = attn_hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = num_layers
        self.batch_first = batch_first
        self.attn_method = attn_method.lower()
        self.coverage = coverage
        self.attn_as_input=attn_as_input
        
        if self.attn_method == 'vanilla':
            self.lstm_ = torch.nn.LSTMCell(
                self.input_size, 
                self.hidden_size
            ).cuda()
        elif self.attn_method[:8] == 'bahdanau':
            self.lstm_ = torch.nn.LSTMCell(
                self.input_size+self.hidden_size, 
                self.hidden_size
            ).cuda()
            self.attn_layer = AttentionBahdanau(
                src_seq_len=self.src_seq_len,
                trg_seq_len=self.trg_seq_len,
                attn_hidden_size=self.attn_hidden_size,
                hidden_size=self.hidden_size,
                attn_method=self.attn_method,
                coverage=self.coverage
            ).cuda()
        elif self.attn_method[:5] == 'luong':
            if self.attn_as_input:
                self.lstm_ = torch.nn.LSTMCell(
                    self.input_size+self.hidden_size, 
                    self.hidden_size
                ).cuda()
            else:
                self.lstm_ = torch.nn.LSTMCell(
                    self.input_size,
                    self.hidden_size
                ).cuda()
            self.attn_layer = AttentionLuong(
                src_seq_len=self.src_seq_len,
                trg_seq_len=self.trg_seq_len,
                attn_hidden_size=self.attn_hidden_size,
                hidden_size=self.hidden_size,
                attn_method=self.attn_method, 
                coverage=self.coverage
            ).cuda()
        
    def forward(self, input_, hidden_, h_attn, encoder_hy, hidden_attn, past_attn):
            
        if self.batch_first:
            input_ = input_.transpose(0,1)

        batch_size = input_.size(1)
        
        output_ = []
        out_attn = []
        if self.attn_method == 'vanilla':
            for k in range(input_.size(0)):
                hidden_ = self.lstm_(input_[k], hidden_)
                output_.append(hidden_[0])
        elif self.attn_method[:8] == 'bahdanau':
            if self.coverage == 'concat' or self.coverage == 'simple':
                for k in range(input_.size(0)):
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_[0], 
                        encoder_hy.transpose(0,1),
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = 0.5*attn + 0.5*past_attn
                    x_input = torch.cat((input_[k], h_attn), 1)
                    hidden_ = self.lstm_(x_input, hidden_)
                    output_.append(hidden_[0])
                    out_attn.append(attn)
            else:
                for k in range(input_.size(0)):
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_[0], 
                        encoder_hy.transpose(0,1),
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = past_attn + attn
                    x_input = torch.cat((input_[k], h_attn), 1)
                    hidden_ = self.lstm_(x_input, hidden_)
                    output_.append(hidden_[0])
                    out_attn.append(attn)
        elif self.attn_method[:5] == 'luong':
            # luong need init h_attn
            if self.coverage == 'concat' or self.coverage == 'simple':
                batch_size = input_.size(1)
                for k in range(input_.size(0)):
                    if self.attn_as_input:
                        x_input = torch.cat((input_[k], h_attn), 1)
                    else:
                        x_input = input_[k]
                    hidden_ = self.lstm_(x_input, hidden_)
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_[0], 
                        encoder_hy.transpose(0,1), 
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = 0.5*attn + 0.5*past_attn
                    output_.append(h_attn)
                    out_attn.append(attn)
            else:
                batch_size = input_.size(1)
                for k in range(input_.size(0)):
                    if self.attn_as_input:
                        x_input = torch.cat((input_[k], h_attn), 1)
                    else:
                        x_input = input_[k]
                    hidden_ = self.lstm_(x_input, hidden_)
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_[0], 
                        encoder_hy.transpose(0,1), 
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = past_attn + attn
                    output_.append(h_attn)
                    out_attn.append(attn)
            
        len_seq = input_.size(0)
        batch_size, hidden_size = output_[0].size()
        output_ = torch.cat(output_, 0).view(
            len_seq, 
            batch_size, 
            hidden_size
        )
        if not self.attn_method == 'vanilla':
            out_attn = torch.cat(out_attn, 0).view(
                len_seq,
                attn.size(0),
                attn.size(1),
            )
        
        if self.batch_first:
            output_ = output_.transpose(0,1)
   
        return output_, hidden_, h_attn, out_attn, hidden_attn, past_attn
'''
GRU decoder
'''
class GRUDecoder(torch.nn.Module):
    def __init__(
        self,
        src_seq_len,
        trg_seq_len,
        attn_hidden_size,
        input_size,
        hidden_size,
        num_layers,
        attn_method,
        coverage,
        attn_as_input,
        batch_first
    ):
        super(GRUDecoder, self).__init__()
        # parameters
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.attn_hidden_size = attn_hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = num_layers
        self.batch_first = batch_first
        self.attn_method = attn_method.lower()
        self.coverage = coverage
        self.attn_as_input=attn_as_input
        
        if self.attn_method == 'vanilla':
            self.gru_ = torch.nn.GRUCell(
                self.input_size, 
                self.hidden_size
            ).cuda()
        elif self.attn_method[:8] == 'bahdanau':
            self.gru_ = torch.nn.GRUCell(
                self.input_size+self.hidden_size, 
                self.hidden_size
            ).cuda()
            self.attn_layer = AttentionBahdanau(
                src_seq_len=self.src_seq_len,
                trg_seq_len=self.trg_seq_len,
                attn_hidden_size=self.attn_hidden_size,
                hidden_size=self.hidden_size,
                attn_method=self.attn_method,
                coverage=self.coverage
            ).cuda()
        elif self.attn_method[:5] == 'luong':
            if self.attn_as_input:
                self.gru_ = torch.nn.GRUCell(
                    self.input_size+self.hidden_size, 
                    self.hidden_size
                ).cuda()
            else:
                self.gru_ = torch.nn.GRUCell(
                    self.input_size, 
                    self.hidden_size
                ).cuda()
            self.attn_layer = AttentionLuong(
                src_seq_len=self.src_seq_len,
                trg_seq_len=self.trg_seq_len,
                attn_hidden_size=self.attn_hidden_size,
                hidden_size=self.hidden_size,
                attn_method=self.attn_method,
                coverage=self.coverage
            ).cuda()
        
    def forward(self, input_, hidden_, h_attn, encoder_hy, hidden_attn, past_attn):
            
        if self.batch_first:
            input_ = input_.transpose(0,1)
            
        batch_size = input_.size(1)
        
        output_ = []
        out_attn = []
        if self.attn_method == 'vanilla':
            for k in range(input_.size(0)):
                hidden_ = self.gru_(input_[k], hidden_)
                output_.append(hidden_)
        elif self.attn_method[:8] == 'bahdanau':
            if self.coverage == 'concat' or self.coverage == 'simple':
                for k in range(input_.size(0)):
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_,
                        encoder_hy.transpose(0,1),
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = 0.5*attn + 0.5*past_attn
                    x_input = torch.cat((input_[k], h_attn), 1)
                    hidden_ = self.gru_(x_input, hidden_)
                    output_.append(hidden_)
                    out_attn.append(attn)
            else:
                for k in range(input_.size(0)):
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_, 
                        encoder_hy.transpose(0,1),
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = past_attn + attn
                    x_input = torch.cat((input_[k], h_attn), 1)
                    hidden_ = self.gru_(x_input, hidden_)
                    output_.append(hidden_)
                    out_attn.append(attn)
        elif self.attn_method[:5] == 'luong':
            if self.coverage == 'concat' or self.coverage == 'simple':
                batch_size = input_.size(1)
                for k in range(input_.size(0)):
                    if self.attn_as_input:
                        x_input = torch.cat((input_[k], h_attn), 1)
                    else:
                        x_input = input_[k]
                    hidden_ = self.gru_(x_input, hidden_)
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_, 
                        encoder_hy.transpose(0,1),
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = 0.5*attn + 0.5*past_attn
                    output_.append(h_attn)
                    out_attn.append(attn)
            else:
                batch_size = input_.size(1)
                for k in range(input_.size(0)):
                    if self.attn_as_input:
                        x_input = torch.cat((input_[k], h_attn), 1)
                    else:
                        x_input = input_[k]
                    hidden_ = self.gru_(x_input, hidden_)
                    h_attn, attn, hidden_attn = self.attn_layer(
                        hidden_, 
                        encoder_hy.transpose(0,1),
                        past_attn=past_attn,
                        hidden_attn=hidden_attn
                    )
                    past_attn = past_attn + attn
                    output_.append(h_attn)
                    out_attn.append(attn)
            
        len_seq = input_.size(0)
        batch_size, hidden_size = output_[0].size()
        output_ = torch.cat(output_, 0).view(
            len_seq, 
            batch_size, 
            hidden_size
        )
        if not self.attn_method == 'vanilla':
            out_attn = torch.cat(out_attn, 0).view(
                len_seq,
                attn.size(0),
                attn.size(1),
            )
        
        if self.batch_first:
            output_ = output_.transpose(0,1)

        return output_, hidden_, h_attn, out_attn, hidden_attn, past_attn
'''
sequence to sequence model
''' 
class Seq2Seq(torch.nn.Module):
    
    def __init__(
        self,
        src_seq_len=400,
        trg_seq_len=100,
        src_emb_dim=128,
        trg_emb_dim=128,
        src_hidden_dim=256,
        trg_hidden_dim=256,
        attn_hidden_dim=256,
        src_vocab_size=999,
        trg_vocab_size=999,
        src_nlayer=2,
        trg_nlayer=1,
        batch_first=True,
        src_bidirect=True,
        dropout=0.0,
        attn_method='vanilla',
        coverage='vanilla',
        network_='gru',
        attn_as_input=True, # For Luong's method only
        shared_emb=True
    ):
        super(Seq2Seq, self).__init__()
        # parameters
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_nlayer = src_nlayer
        self.trg_nlayer = trg_nlayer
        self.batch_first = batch_first
        self.src_bidirect = src_bidirect
        self.dropout = dropout
        self.attn_method = attn_method.lower()
        self.coverage = coverage.lower()
        self.network_ = network_.lower()
        self.shared_emb=shared_emb
        self.attn_as_input=attn_as_input
        
        self.src_num_directions = 1
        if self.src_bidirect:
            self.src_hidden_dim = src_hidden_dim // 2
            self.src_num_directions = 2
        
        # source embedding and target embedding
        # the same for summarization.
        if self.shared_emb:
            self.embedding = torch.nn.Embedding(
                self.src_vocab_size,
                self.src_emb_dim,
                padding_idx=0
            ).cuda()
            torch.nn.init.uniform(self.embedding.weight, -1.0, 1.0)
        else:
            self.src_embedding = torch.nn.Embedding(
                self.src_vocab_size,
                self.src_emb_dim,
                padding_idx=0
            ).cuda()
            torch.nn.init.uniform(self.src_embedding.weight, -1.0, 1.0)

            self.trg_embedding = torch.nn.Embedding(
                self.src_vocab_size,
                self.src_emb_dim,
                padding_idx=0
            ).cuda()
            torch.nn.init.uniform(self.trg_embedding.weight, -1.0, 1.0)
        # choose network
        if self.network_ == 'lstm':
            # encoder
            self.encoder = torch.nn.LSTM(
                input_size=self.src_emb_dim,
                hidden_size=self.src_hidden_dim,
                num_layers=self.src_nlayer,
                batch_first=self.batch_first,
                dropout=self.dropout,
                bidirectional=self.src_bidirect
            ).cuda()
            # decoder
            self.decoder = LSTMDecoder(
                src_seq_len=self.src_seq_len,
                trg_seq_len=self.trg_seq_len,
                attn_hidden_size=self.attn_hidden_dim,
                input_size=self.trg_emb_dim,
                hidden_size=self.trg_hidden_dim,
                num_layers=self.trg_nlayer,
                attn_method=self.attn_method,
                coverage=self.coverage,
                attn_as_input=self.attn_as_input,
                batch_first=self.batch_first
            ).cuda()
        elif self.network_ == 'gru':
            # encoder
            self.encoder = torch.nn.GRU(
                input_size=self.src_emb_dim,
                hidden_size=self.src_hidden_dim,
                num_layers=self.src_nlayer,
                batch_first=self.batch_first,
                dropout=self.dropout,
                bidirectional=self.src_bidirect
            ).cuda()
            # decoder
            self.decoder = GRUDecoder(
                src_seq_len=self.src_seq_len,
                trg_seq_len=self.trg_seq_len,
                attn_hidden_size=self.attn_hidden_dim,
                input_size=self.trg_emb_dim,
                hidden_size=self.trg_hidden_dim,
                num_layers=self.trg_nlayer,
                attn_method=self.attn_method,
                coverage=self.coverage,
                attn_as_input=self.attn_as_input,
                batch_first=self.batch_first
            ).cuda()
        # encoder to decoder
        self.encoder2decoder = torch.nn.Linear(
            self.src_hidden_dim*self.src_num_directions,
            self.trg_hidden_dim
        ).cuda()
        # decoder to vocab
        self.decoder2vocab = torch.nn.Linear(
            self.trg_hidden_dim,
            self.trg_vocab_size
        ).cuda()
        
    def forward(self, input_src, input_trg):
        if self.shared_emb:
            src_emb = self.embedding(input_src)
            trg_emb = self.embedding(input_trg)
        else:
            src_emb = self.src_embedding(input_src)
            trg_emb = self.trg_embedding(input_trg)
            
        batch_size = input_src.size(1)
        if self.batch_first:
            batch_size = input_src.size(0)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers*self.src_num_directions,
            batch_size, 
            self.src_hidden_dim
        )).cuda()
        hidden_attn = Variable(torch.zeros(
            batch_size,
            self.attn_hidden_dim
        )).cuda()
        past_attn = Variable(torch.zeros(
            batch_size, 
            self.src_seq_len
        )).cuda()
        h_attn = Variable(
            torch.FloatTensor(torch.zeros(batch_size, self.trg_hidden_dim))
        ).cuda()

        if self.network_ == 'lstm':
            c0_encoder = Variable(torch.zeros(
                self.encoder.num_layers*self.src_num_directions,
                batch_size, self.src_hidden_dim)).cuda()

            src_h, (src_h_t, src_c_t) = self.encoder(
                src_emb, 
                (h0_encoder, c0_encoder)
            )

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
                c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                c_t = src_c_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
            decoder_c0 = c_t
        
            encoder_hy = src_h.transpose(0,1)
        
            trg_h, (_, _), _, attn_, hidden_attn, _ = self.decoder(
                trg_emb,
                (decoder_h0, decoder_c0),
                h_attn,
                encoder_hy,
                hidden_attn,
                past_attn
            )
        elif self.network_ == 'gru':
            src_h, src_h_t = self.encoder(
                src_emb,
                h0_encoder
            )

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
        
            encoder_hy = src_h.transpose(0,1)
        
            trg_h, _, _, attn_, hidden_attn, _ = self.decoder(
                trg_emb,
                decoder_h0,
                h_attn,
                encoder_hy,
                hidden_attn,
                past_attn
            )
        # prepare output
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )
        # here consume a lot of memory. output
        # decoder_ouput is also logits in this code.
        decoder_output = self.decoder2vocab(trg_h_reshape)
        decoder_output = decoder_output.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_output.size(1)
        )

        return decoder_output, attn_
    
    def forward_encoder(self, input_src):
        if self.shared_emb:
            src_emb = self.embedding(input_src)
        else:
            src_emb = self.src_embedding(input_src)
            
        batch_size = input_src.size(1)
        if self.batch_first:
            batch_size = input_src.size(0)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers*self.src_num_directions,
            batch_size, 
            self.src_hidden_dim
        )).cuda()
        hidden_attn = Variable(torch.zeros(
            batch_size,
            self.attn_hidden_dim
        )).cuda()
        past_attn = Variable(torch.zeros(
            batch_size, 
            self.src_seq_len
        )).cuda()
        h_attn = Variable(
            torch.FloatTensor(torch.zeros(batch_size, self.trg_hidden_dim))
        ).cuda()

        if self.network_ == 'lstm':
            c0_encoder = Variable(torch.zeros(
                self.encoder.num_layers*self.src_num_directions,
                batch_size, self.src_hidden_dim)).cuda()

            src_h, (src_h_t, src_c_t) = self.encoder(
                src_emb, 
                (h0_encoder, c0_encoder)
            )

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
                c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                c_t = src_c_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
            decoder_c0 = c_t
        
            encoder_hy = src_h.transpose(0,1)
            
            return encoder_hy, (decoder_h0, decoder_c0), hidden_attn, past_attn
        
        elif self.network_ == 'gru':
            src_h, src_h_t = self.encoder(
                src_emb,
                h0_encoder
            )

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
        
            encoder_hy = src_h.transpose(0,1)
        
            return encoder_hy, decoder_h0, h_attn, hidden_attn, past_attn
    
    def forward_onestep_decoder(
        self, 
        input_trg,
        hidden_decoder,
        h_attn,
        encoder_hy,
        hidden_attn,
        past_attn
    ):
        if self.shared_emb:
            trg_emb = self.embedding(input_trg)
        else:
            trg_emb = self.trg_embedding(input_trg)
            
        batch_size = input_trg.size(1)
        if self.batch_first:
            batch_size = input_trg.size(0)

        if self.network_ == 'lstm':
            trg_h, hidden_decoder, h_attn, attn_, hidden_attn, past_attn = self.decoder(
                trg_emb,
                hidden_decoder,
                h_attn,
                encoder_hy,
                hidden_attn,
                past_attn
            )
        elif self.network_ == 'gru':
            trg_h, hidden_decoder, h_attn, attn_, hidden_attn, past_attn = self.decoder(
                trg_emb,
                hidden_decoder,
                h_attn,
                encoder_hy,
                hidden_attn,
                past_attn
            )
        # prepare output
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )
        # here consume a lot of memory. output
        # decoder_ouput is also logits in this code.
        decoder_output = self.decoder2vocab(trg_h_reshape)
        decoder_output = decoder_output.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_output.size(1)
        )

        return decoder_output, hidden_decoder, h_attn, hidden_attn, past_attn
    
    def decode(self, logits):
        # here consume a lot of memory.
        word_probs = F.softmax(
            logits.view(-1, logits.size(2)), 
            dim=1
        )
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )

        return word_probs
    
