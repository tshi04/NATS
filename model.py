import torch
import torch.nn.functional as F
from torch.autograd import Variable
'''
Bahdanau, D., Cho, K., & Bengio, Y. (2014). 
Neural machine translation by jointly learning to align and translate. 
arXiv preprint arXiv:1409.0473.
'''
class AttentionBahdanau(torch.nn.Module):

    def __init__(
        self,
        attn_method='bahdanau_dot',
        hidden_size=100,
        bias=True
    ):
        super(AttentionBahdanau, self).__init__()   
        
        self.attn_method = attn_method.lower()
        self.hidden_size = hidden_size
        self.bias = bias

        self.softmax_ = torch.nn.Softmax().cuda()
        self.tanh_ = torch.nn.Tanh().cuda()

        if self.attn_method == 'bahdanau_concat':
            self.attn_in = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hidden_size*2,
                    self.hidden_size,
                    bias=self.bias
                ),
                torch.nn.Linear(self.hidden_size, 1, bias=self.bias)
            ).cuda()
        
    def forward(self, last_dehy, enhy):
        dehy_new = last_dehy.unsqueeze(2)

        if self.attn_method == 'bahdanau_concat':
            dehy_rep = last_dehy.unsqueeze(1)
            dehy_rep = dehy_rep.repeat(1, enhy.size(1), 1)
            cat_hy = torch.cat((enhy, dehy_rep), 2)
            attn = self.attn_in(cat_hy).squeeze(2)
        else:
            attn = torch.bmm(enhy, dehy_new).squeeze(2)

        attn = self.softmax_(attn)
        attn2 = attn.view(attn.size(0), 1, attn.size(1))
        h_attn = torch.bmm(attn2, enhy).squeeze(1)
        h_attn = self.tanh_(h_attn)

        return h_attn, attn
'''
Luong, M. T., Pham, H., & Manning, C. D. (2015). 
Effective approaches to attention-based neural machine translation. 
arXiv preprint arXiv:1508.04025.
'''
class AttentionLuong(torch.nn.Module):
    
    def __init__(
        self,
        attn_method='luong_dot',
        hidden_size=100,
        bias=False
    ):
        super(AttentionLuong, self).__init__()
        self.method = attn_method.lower()
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.softmax_ = torch.nn.Softmax().cuda()
        self.tanh_ = torch.nn.Tanh().cuda()
        
        if self.method == 'luong_concat':
            self.attn_in = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hidden_size*2,
                    self.hidden_size,
                    bias=self.bias
                ),
                torch.nn.Linear(self.hidden_size, 1, bias=self.bias)
            ).cuda()
        else:
            if self.method == 'luong_general':
                self.attn_in = torch.nn.Linear(
                    self.hidden_size, 
                    self.hidden_size,
                    bias=self.bias
                ).cuda()
                
        self.attn_out = torch.nn.Linear(
            self.hidden_size*2,
            self.hidden_size,
            bias=self.bias
        ).cuda()
        
    def forward(self, dehy, enhy):
        dehy_new = dehy.unsqueeze(2)
        enhy_new = enhy
        
        if self.method == 'luong_concat':
            dehy_rep = dehy.unsqueeze(1)
            dehy_rep = dehy_rep.repeat(1, enhy.size(1), 1)
            cat_hy = torch.cat((enhy, dehy_rep), 2)
            attn = self.attn_in(cat_hy).squeeze(2)
        else:
            if self.method == 'luong_general':
                enhy_new = self.attn_in(enhy)
        
            attn = torch.bmm(enhy_new, dehy_new).squeeze(2)
        
        attn = self.softmax_(attn)
        attn2 = attn.view(attn.size(0), 1, attn.size(1))

        attn_enhy = torch.bmm(attn2, enhy_new).squeeze(1)
        
        h_attn = self.attn_out(torch.cat((attn_enhy, dehy), 1))
        h_attn = self.tanh_(h_attn)

        return h_attn, attn

'''
LSTM decoder
'''    
class LSTMDecoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        attn_method='bahdanau_dot',
        batch_first=True
    ):
        super(LSTMDecoder, self).__init__()
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = num_layers
        self.batch_first = batch_first
        self.attn_method = attn_method.lower()
        
        self.softmax_ = torch.nn.Softmax().cuda()
        self.tanh_ = torch.nn.Tanh().cuda()
        self.sigmoid_ = torch.nn.Sigmoid().cuda()
        
        if self.attn_method == 'vanilla':
            self.lstm_ = torch.nn.LSTMCell(
                self.input_size, 
                self.hidden_size
            )
        elif self.attn_method[:8] == 'bahdanau':
            self.lstm_ = torch.nn.LSTMCell(
                self.input_size+self.hidden_size, 
                self.hidden_size
            )
            self.attn_layer = AttentionBahdanau(
                attn_method=self.attn_method,
                hidden_size=self.hidden_size
            ).cuda()
            
        else:
            self.lstm_ = torch.nn.LSTMCell(
                self.input_size+self.hidden_size, 
                self.hidden_size
            )
            self.attn_layer = AttentionLuong(
                attn_method=self.attn_method, 
                hidden_size=self.hidden_size
            ).cuda()
        
    def forward(self, input_, hidden_, encoder_hy):
            
        if self.batch_first:
            input_ = input_.transpose(0,1)
            
        output_ = []
        if self.attn_method == 'vanilla':
            for k in range(input_.size(0)):
                hidden_ = self.lstm_(input_[k], hidden_)
                output_.append(hidden_[0])
                
        elif self.attn_method[:8] == 'bahdanau':
            for k in range(input_.size(0)):
                h_attn, attn = self.attn_layer(hidden_[0], encoder_hy.transpose(0,1))
                x_input = torch.cat((input_[k], h_attn), 1)
                hidden_ = self.lstm_(x_input, hidden_)
                output_.append(hidden_[0])
        else:
            batch_size = input_.size(1)
            h_attn = Variable(
                torch.FloatTensor(torch.zeros(batch_size, self.hidden_size))
            ).cuda()
            for k in range(input_.size(0)):
                x_input = torch.cat((input_[k], h_attn), 1)
                hidden_ = self.lstm_(x_input, hidden_)
                h_attn, attn = self.attn_layer(hidden_[0], encoder_hy.transpose(0,1))
                output_.append(hidden_[0])
            
        len_seq = input_.size(0)
        batch_size, hidden_size = output_[0].size()
        output_ = torch.cat(output_, 0).view(
            len_seq, 
            batch_size, 
            hidden_size
        )
        
        if self.batch_first:
            output_ = output_.transpose(0,1)
            
        return output_, hidden_, attn

'''
GRU decoder
'''
class GRUDecoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        attn_method='bahdanau_dot',
        batch_first=True
    ):
        super(GRUDecoder, self).__init__()
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = num_layers
        self.batch_first = batch_first
        self.attn_method = attn_method.lower()
        
        self.softmax_ = torch.nn.Softmax().cuda()
        self.tanh_ = torch.nn.Tanh().cuda()
        self.sigmoid_ = torch.nn.Sigmoid().cuda()
        
        if self.attn_method == 'vanilla':
            self.gru_ = torch.nn.GRUCell(
                self.input_size, 
                self.hidden_size
            )
        elif self.attn_method[:8] == 'bahdanau':
            self.gru_ = torch.nn.GRUCell(
                self.input_size+self.hidden_size, 
                self.hidden_size
            )
            self.attn_layer = AttentionBahdanau(
                attn_method=self.attn_method,
                hidden_size=self.hidden_size
            ).cuda()
            
        else:
            self.gru_ = torch.nn.GRUCell(
                self.input_size+self.hidden_size, 
                self.hidden_size
            )
            self.attn_layer = AttentionLuong(
                attn_method=self.attn_method, 
                hidden_size=self.hidden_size
            ).cuda()
        
    def forward(self, input_, hidden_, encoder_hy):
            
        if self.batch_first:
            input_ = input_.transpose(0,1)
            
        output_ = []
        if self.attn_method == 'vanilla':
            for k in range(input_.size(0)):
                hidden_ = self.gru_(input_[k], hidden_)
                output_.append(hidden_)
                
        elif self.attn_method[:8] == 'bahdanau':
            for k in range(input_.size(0)):
                h_attn, attn = self.attn_layer(hidden_, encoder_hy.transpose(0,1))
                x_input = torch.cat((input_[k], h_attn), 1)
                hidden_ = self.gru_(x_input, hidden_)
                output_.append(hidden_)
        else:
            batch_size = input_.size(1)
            h_attn = Variable(
                torch.FloatTensor(torch.zeros(batch_size, self.hidden_size))
            ).cuda()
            for k in range(input_.size(0)):
                x_input = torch.cat((input_[k], h_attn), 1)
                hidden_ = self.gru_(x_input, hidden_)
                h_attn, attn = self.attn_layer(hidden_, encoder_hy.transpose(0,1))
                output_.append(hidden_)
            
        len_seq = input_.size(0)
        batch_size, hidden_size = output_[0].size()
        output_ = torch.cat(output_, 0).view(
            len_seq, 
            batch_size, 
            hidden_size
        )
        
        if self.batch_first:
            output_ = output_.transpose(0,1)
            
        return output_, hidden_, attn

    
class Seq2Seq(torch.nn.Module):
    
    def __init__(
        self,
        src_emb_dim=100,
        trg_emb_dim=100,
        src_hidden_dim=50,
        trg_hidden_dim=50,
        src_vocab_size=999,
        trg_vocab_size=999,
        src_nlayer=1,
        trg_nlayer=1,
        batch_first=True,
        src_bidirect=True,
        dropout=0.0,
        attn_method='vanilla',
        network_='gru'
    ):
        super(Seq2Seq, self).__init__()
        # parameters
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_nlayer = src_nlayer
        self.trg_nlayer = trg_nlayer
        self.batch_first = batch_first
        self.src_bidirect = src_bidirect
        self.dropout = dropout
        self.attn_method = attn_method
        self.network_ = network_.lower()
        
        self.softmax_ = torch.nn.Softmax().cuda()
        self.tanh_ = torch.nn.Tanh().cuda()
        self.sigmoid_ = torch.nn.Sigmoid().cuda()
        
        self.src_num_directions = 1
        if self.src_bidirect:
            self.src_hidden_dim = src_hidden_dim // 2
            self.src_num_directions = 2
        
        # source embedding and target embedding
        # the same for summarization.
        self.embedding = torch.nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            padding_idx=0
        ).cuda()
        torch.nn.init.uniform(self.embedding.weight, -1.0, 1.0)
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
                input_size=self.trg_emb_dim,
                hidden_size=self.trg_hidden_dim,
                batch_first=self.batch_first,
                attn_method=self.attn_method
            ).cuda()
        else:
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
                input_size=self.trg_emb_dim,
                hidden_size=self.trg_hidden_dim,
                batch_first=self.batch_first,
                attn_method=self.attn_method
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
        src_emb = self.embedding(input_src)
        trg_emb = self.embedding(input_trg)
        
        batch_size = input_src.size(1)
        if self.batch_first:
            batch_size = input_src.size(0)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers*self.src_num_directions,
            batch_size, self.src_hidden_dim)).cuda()
        
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
            decoder_h0 = self.tanh_(decoder_h0)
            decoder_c0 = c_t
        
            encoder_hy = src_h.transpose(0,1)
        
            trg_h, (_, _), self.attn_ = self.decoder(
                trg_emb,
                (decoder_h0, decoder_c0),
                encoder_hy
            )
        
        else:
            src_h, src_h_t = self.encoder(
                src_emb, 
                h0_encoder
            )

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = self.tanh_(decoder_h0)
        
            encoder_hy = src_h.transpose(0,1)
        
            trg_h, _, self.attn_ = self.decoder(
                trg_emb,
                decoder_h0,
                encoder_hy
            )
        
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )
                
        decoder_output = self.decoder2vocab(trg_h_reshape)
        decoder_output = decoder_output.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_output.size(1)
        )

        return decoder_output, self.attn_
    
    def decode(self, logits):
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = self.softmax_(logits_reshape)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )

        return word_probs
    
