import torch
from torch.autograd import Variable
'''
Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
'''
class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        
    def forward(self, hy, encoder_hy):
        target = hy.unsqueeze(2)
        attn = torch.bmm(encoder_hy, target).squeeze(2)
        attn = torch.nn.Softmax()(attn)
        attn2 = attn.view(attn.size(0), 1, attn.size(1))
        
        weighted_hy = torch.bmm(attn2, encoder_hy).squeeze(1)
        
        h_attn = weighted_hy + hy
        h_attn = torch.nn.Tanh()(h_attn)
        
        return h_attn, attn

class LSTMAttention(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=False
    ):
        super(LSTMAttention, self).__init__()
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = num_layers
        self.batch_first = batch_first
        
        self.lstm_input_w = torch.nn.Linear(
            self.input_size,
            self.hidden_size*4
        ).cuda()
        self.lstm_hidden_w = torch.nn.Linear(
            self.hidden_size,
            self.hidden_size*4
        ).cuda()
        
        self.attn_layer = Attention().cuda()
        
    def forward(self, input_, hidden_, encoder_hy):
        # user defined lstm with attention
        def lstm_attn(input_, hidden_, encoder_hy):
            hx, cx = hidden_
            gates = self.lstm_input_w(input_) + self.lstm_hidden_w(hx)
            ingate, cellgate, forgetgate, outgate = gates.chunk(4,1)
            
            ingate = torch.nn.Sigmoid()(ingate)
            forgetgate = torch.nn.Sigmoid()(forgetgate)
            outgate = torch.nn.Sigmoid()(outgate)
            cellgate = torch.nn.Tanh()(cellgate)
            
            cy = forgetgate*cx + ingate*cellgate
            hy = outgate*torch.nn.Tanh()(cy)
            
            h_attn, attn = self.attn_layer(hy, encoder_hy.transpose(0,1))
            
            return h_attn, cy
        
        if self.batch_first:
            input_ = input_.transpose(0,1)
            
        output_ = []
        for k in range(input_.size(0)):
            hidden_ = lstm_attn(input_[k], hidden_, encoder_hy)
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
            
        return output_, hidden_

class seq2seqAttn(torch.nn.Module):
    
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
        batch_first=True,
        src_bidirect=True,
        batch_size=128,
        dropout=0.0
    ):
        super(seq2seqAttn, self).__init__()
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
        self.batch_size = batch_size
        self.dropout = dropout
        
        self.src_num_directions = 1
        if self.src_bidirect:
            self.src_hidden_dim = src_hidden_dim // 2
            self.src_num_directions = 2
        # embedding
        self.embedding = torch.nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            padding_idx=0
        ).cuda()
        torch.nn.init.uniform(self.embedding.weight, -1.0, 1.0)
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
        self.decoder = LSTMAttention(
            input_size=self.trg_emb_dim,
            hidden_size=self.trg_hidden_dim,
            batch_first=self.batch_first
        ).cuda()
        # encoder to decoder
        self.encoder2decoder = torch.nn.Linear(
            self.src_hidden_dim*self.src_num_directions,
            self.trg_hidden_dim
        ).cuda()
        torch.nn.init.constant(self.encoder2decoder.bias, 0.0)
        # decoder to vocab
        self.decoder2vocab = torch.nn.Linear(
            self.trg_hidden_dim,
            self.trg_vocab_size
        ).cuda()
        torch.nn.init.constant(self.decoder2vocab.bias, 0.0)
        
    def forward(self, input_src, input_trg):
        
        src_emb = self.embedding(input_src)
        trg_emb = self.embedding(input_trg)
        
        batch_size = input_src.size(1)
        if self.batch_first:
            batch_size = input_src.size(0)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers*self.src_num_directions,
            self.batch_size,
            self.src_hidden_dim
        ), requires_grad=False).cuda()
        
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers*self.src_num_directions,
            self.batch_size,
            self.src_hidden_dim
        ), requires_grad=False).cuda()

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
        decoder_h0 = torch.nn.Tanh()(decoder_h0)
        decoder_c0 = c_t
        
        encoder_hy = src_h.transpose(0,1)
        
        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_h0, decoder_c0),
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
        
        return decoder_output

    def decode(self, logits):
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = torch.nn.Softmax()(logits_reshape)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )

        return word_probs
