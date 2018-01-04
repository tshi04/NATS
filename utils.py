import numpy as np
import torch
import time
from torch.autograd import Variable
'''
fast beam search
'''
def tensor_transformer(seq0, batch_size, beam_size):
    seq = seq0.unsqueeze(2)
    seq = seq.repeat(1, 1, beam_size, 1)
    seq = seq.contiguous().view(batch_size, beam_size*beam_size, seq.size(3))
    return seq

def fast_beam_search(
    model, 
    src_text,
    vocab2id,
    beam_size=4, 
    max_len=20
):
    batch_size = src_text.size(0)
    src_text_rep = src_text.unsqueeze(1).clone().repeat(1, beam_size, 1).view(-1, src_text.size(1)).cuda()
    encoder_hy, hidden_decoder_new, h_attn_new, hidden_attn_new, past_attn_new = model.forward_encoder(src_text_rep)
    
    beam_seq = Variable(torch.LongTensor(
        batch_size, beam_size, max_len+1).fill_(vocab2id['<pad>'])).cuda()
    beam_seq[:, :, 0] = vocab2id['<s>']
    beam_prb = torch.FloatTensor(batch_size, beam_size).fill_(0.0)
    last_wd = Variable(torch.LongTensor(batch_size, beam_size, 1).fill_(vocab2id['<s>'])).cuda()
    
    for j in range(max_len):
        logits, hidden_decoder, h_attn, hidden_attn, past_attn = model.forward_onestep_decoder(
            last_wd.view(-1, 1), 
            hidden_decoder_new,
            h_attn_new, 
            encoder_hy, 
            hidden_attn_new, 
            past_attn_new)
        prob, wds = model.decode(logits=logits).data.topk(k=beam_size)
        prob = prob.view(batch_size, beam_size, prob.size(1), prob.size(2))
        wds = wds.view(batch_size, beam_size, wds.size(1), wds.size(2))
        if j == 0:
            beam_prb = prob[:, 0, 0]
            beam_seq[:, :, 1] = wds[:, 0, 0]
            last_wd = Variable(wds[:, 0, 0].unsqueeze(2).clone()).cuda()
            
            hidden_decoder_new = hidden_decoder
            h_attn_new = h_attn
            hidden_attn_new = hidden_attn
            past_attn_new = past_attn
            continue

        cand_seq = tensor_transformer(beam_seq, batch_size, beam_size)
        cand_seq[:, :, j+1] = wds.squeeze(2).view(batch_size, -1)
        cand_last_wd = wds.squeeze(2).view(batch_size, -1)

        cand_prob = beam_prb.unsqueeze(1).repeat(1, beam_size, 1).transpose(1,2)
        cand_prob += prob[:, :, 0]
        cand_prob = cand_prob.contiguous().view(batch_size, beam_size*beam_size)
        
        hidden_decoder_new = hidden_decoder_new.view(batch_size, beam_size, hidden_decoder_new.size(-1))
        h_attn_new = h_attn_new.view(batch_size, beam_size, h_attn_new.size(1))
        hidden_attn_new = hidden_attn_new.view(batch_size, beam_size, hidden_attn_new.size(1))
        past_attn_new = past_attn_new.view(batch_size, beam_size, past_attn_new.size(1))
        
        hidden_decoder = hidden_decoder.view(batch_size, beam_size, hidden_decoder.size(-1))
        hidden_decoder = tensor_transformer(hidden_decoder, batch_size, beam_size)
        h_attn = h_attn.view(batch_size, beam_size, h_attn.size(1))
        h_attn = tensor_transformer(h_attn, batch_size, beam_size)
        hidden_attn = hidden_attn.view(batch_size, beam_size, hidden_attn.size(1))
        hidden_attn = tensor_transformer(hidden_attn, batch_size, beam_size)
        past_attn = past_attn.view(batch_size, beam_size, past_attn.size(1))
        past_attn = tensor_transformer(past_attn, batch_size, beam_size)
        
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size, dim=1)
        for x in range(batch_size):
            for b in range(beam_size):
                last_wd[x, b]  = cand_last_wd[x, tmp_idx[x, b]]
                beam_seq[x, b] = cand_seq[x, tmp_idx[x, b]]
                beam_prb[x, b] = tmp_prb[x, b]
                
                hidden_decoder_new[x, b] = hidden_decoder[x, tmp_idx[x, b]]
                h_attn_new[x, b] = h_attn[x, tmp_idx[x, b]]
                hidden_attn_new[x, b] = hidden_attn[x, tmp_idx[x, b]]
                past_attn_new[x, b] = past_attn[x, tmp_idx[x, b]]
             
        hidden_decoder_new = hidden_decoder_new.view(-1, hidden_decoder_new.size(-1))
        h_attn_new = h_attn_new.view(-1, h_attn_new.size(-1))
        hidden_attn_new = hidden_attn_new.view(-1, hidden_attn_new.size(-1))
        past_attn_new = past_attn_new.view(-1, past_attn_new.size(-1))
        
    return beam_seq, beam_prb
'''
can handle batches.
still very slow.
'''
def batch_beam_search(
    model, 
    src_text,
    vocab2id,
    beam_size=4, 
    max_len=20
):
    max_len += 1
    batch_size = src_text.size(0)
    beam_seq = Variable(torch.LongTensor(
        batch_size, beam_size, max_len).fill_(vocab2id['<pad>'])).cuda()
    beam_seq[:, :, 0] = vocab2id['<s>']
    beam_prb = torch.FloatTensor(batch_size, beam_size).fill_(0.0)
    src_text_rep = src_text.unsqueeze(1).clone().repeat(1, beam_size, 1).cuda()
    for j in range(max_len-1):
        logits, _ = model(
            src_text_rep.view(-1, src_text_rep.size(2)),
            beam_seq.view(-1, beam_seq.size(2))
        )
        prob, wds = model.decode(logits=logits).data.topk(k=beam_size)
        prob = prob.view(batch_size, beam_size, prob.size(1), prob.size(2))
        wds = wds.view(batch_size, beam_size, wds.size(1), wds.size(2))
        if j == 0:
            beam_prb = prob[:, 0, 0]
            beam_seq[:, :, 1] = wds[:, 0, 0]
            continue
        
        if j == 2:
            print beam_seq
            print beam_seq.view(-1, beam_seq.size(2))[:, j]
            print prob[:, :, j]
            break
        cand_seq = beam_seq.repeat(1, beam_size, 1)
        cand_seq = cand_seq.view(batch_size, beam_size, beam_size, beam_seq.size(2))
        cand_seq = cand_seq.transpose(1,2)
        cand_seq = cand_seq.contiguous().view(batch_size, beam_size*beam_size, cand_seq.size(3))
        cand_seq[:, :,j+1] = wds[:, :, j].contiguous().view(-1)
        
        cand_prob = beam_prb.unsqueeze(1).repeat(1, beam_size, 1).transpose(1,2)
        cand_prob += prob[:, :, j]
        cand_prob = cand_prob.contiguous().view(batch_size, beam_size*beam_size)
        
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size, dim=1)
        for x in range(batch_size):
            for b in range(beam_size):
                beam_seq[x, b] = cand_seq[x, tmp_idx[x, b]]
                beam_prb[x, b] = tmp_prb[x, b]
    return      
    return beam_seq, beam_prb


'''
very old approach
very slow
'''
def beam_search(
    model, 
    src_text,
    vocab2id,
    beam_size=5, 
    max_len=20
):
    max_len += 1
    beam_seq = Variable(torch.LongTensor(beam_size, max_len).fill_(vocab2id['<pad>'])).cuda()
    beam_seq[:,0] = vocab2id['<s>']
    beam_prb = torch.FloatTensor(beam_size).fill_(0.0)
    src_text_rep = src_text.clone().repeat(beam_size, 1).cuda()
    for j in range(max_len-1):
        logits, _ = model(src_text_rep, beam_seq)
        word_prob = model.decode(logits=logits)
        prob, wds = word_prob.topk(k=beam_size)
        if j == 0:
            beam_prb = prob.data[0][0]
            beam_seq[:,1] = wds.data[0][0]
            continue

        cand_seq = beam_seq.repeat(beam_size, 1)
        cand_seq = cand_seq.view(beam_size, beam_size, beam_seq.size(1))
        cand_seq = cand_seq.transpose(0,1)
        cand_seq = cand_seq.contiguous().view(-1, cand_seq.size(2))
        cand_seq[:,j+1] = wds.data[:,j].contiguous().view(-1)
        
        cand_prob = beam_prb.repeat(beam_size, 1).transpose(0,1)
        cand_prob += prob[:,j].data
        cand_prob = cand_prob.contiguous().view(-1)
        
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size)
        for b in range(beam_size):
            beam_seq[b] = cand_seq[tmp_idx[b]]
            beam_prb[b] = tmp_prb[b]
    
    return beam_seq, beam_prb